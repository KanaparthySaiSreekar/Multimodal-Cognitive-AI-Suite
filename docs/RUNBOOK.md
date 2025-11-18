# Operational Runbook: Multimodal Intelligence Workflow Suite

**Version**: 1.0
**Last Updated**: November 2025
**Owner**: ML Platform Team
**Escalation**: ml-platform@company.com

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Common Operational Tasks](#common-operational-tasks)
4. [Incident Response Playbooks](#incident-response-playbooks)
5. [Rollback Procedures](#rollback-procedures)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Contact Information](#contact-information)

---

## 1. Overview

### System Description

The Multimodal Intelligence Workflow Suite provides:
- Document classification (BERT-based)
- Image recognition (Vision Transformer)
- Multimodal fusion (text + image)

**Production Environment**: AWS ECS (us-east-1)
**Deployment**: Blue/green with canary rollout
**SLOs**:
- P95 latency < 600ms
- Error rate < 0.5%
- Availability > 99.5%

### Key URLs

| Environment | Endpoint | Metrics | Logs |
|-------------|----------|---------|------|
| Production | https://api.company.com/v1 | http://grafana.company.com/d/multimodal | CloudWatch Logs |
| Staging | https://api-staging.company.com/v1 | http://grafana-staging.company.com | CloudWatch Logs |

### Quick Reference Commands

```bash
# Check system health
curl https://api.company.com/health

# View recent logs
aws logs tail /aws/ecs/multimodal-prod --follow

# Get current deployment
aws ecs describe-services --cluster multimodal-prod --services multimodal-api

# Check current model version
curl https://api.company.com/model-info
```

---

## 2. System Architecture

### Components

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   ALB       │────▶│   ECS Tasks  │────▶│   Models    │
│ (Load Bal.) │     │  (Auto-scale)│     │ (S3 Storage)│
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
      │                     ▼                     ▼
      │              ┌──────────────┐     ┌─────────────┐
      │              │  CloudWatch  │     │   Drift     │
      └─────────────▶│   Metrics    │     │  Monitor    │
                     └──────────────┘     └─────────────┘
```

### Data Flow

1. **Request**: Client → ALB → ECS Task
2. **Preprocessing**: Text/Image preprocessing
3. **Inference**: Model prediction
4. **Response**: JSON result → Client
5. **Monitoring**: Metrics → CloudWatch/Prometheus

### Dependencies

- **Critical**: S3 (model storage), ECS, ALB
- **Important**: CloudWatch, Secrets Manager
- **Optional**: Grafana, PagerDuty

---

## 3. Common Operational Tasks

### 3.1 Deploy New Model Version

**When**: After model retraining and validation

```bash
# 1. Upload new model to S3
aws s3 cp models/document_classifier_v2.pt \
  s3://multimodal-models-prod/document_classifier/v2/model.pt

# 2. Update model version in config
aws ssm put-parameter \
  --name /multimodal/prod/document_classifier_version \
  --value "v2" \
  --overwrite

# 3. Trigger canary deployment
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --force-new-deployment \
  --deployment-configuration "minimumHealthyPercent=100,maximumPercent=200"

# 4. Monitor canary metrics (first 10% of traffic)
# Watch: https://grafana.company.com/d/multimodal/canary

# 5. If metrics good, promote to 100%
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --desired-count 5

# 6. Verify deployment
curl https://api.company.com/model-info
```

**Validation Checklist**:
- [ ] P95 latency < 600ms
- [ ] Error rate < 0.5%
- [ ] No drift alerts triggered
- [ ] Model accuracy maintained
- [ ] All health checks passing

### 3.2 Scale Service

**When**: Traffic increase or performance degradation

```bash
# Manual scaling
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --desired-count 10

# Check autoscaling status
aws application-autoscaling describe-scalable-targets \
  --service-namespace ecs \
  --resource-id service/multimodal-prod/multimodal-api

# Update autoscaling limits
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/multimodal-prod/multimodal-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 20
```

### 3.3 Update Configuration

**When**: Changing model parameters or system settings

```bash
# View current config
aws ssm get-parameters-by-path \
  --path /multimodal/prod/ \
  --recursive

# Update specific parameter
aws ssm put-parameter \
  --name /multimodal/prod/max_batch_size \
  --value "32" \
  --overwrite

# Restart tasks to pick up new config
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --force-new-deployment
```

### 3.4 Check System Health

```bash
# Health check endpoint
curl https://api.company.com/health

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "document_classifier": "healthy",
#     "image_classifier": "healthy",
#     "multimodal_fusion": "healthy"
#   },
#   "version": "1.2.0"
# }

# Detailed metrics
curl https://api.company.com/metrics | grep -E "(latency|error|drift)"

# Check logs for errors
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "ERROR" \
  --start-time $(date -u -d '1 hour ago' +%s)000
```

---

## 4. Incident Response Playbooks

### 4.1 High Latency (P95 > 1000ms)

**Symptoms**: Slow responses, user complaints, latency alerts

**Diagnosis**:
```bash
# 1. Check current latency
curl https://api.company.com/metrics | grep latency_seconds

# 2. Check resource utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=multimodal-api \
  --start-time $(date -u -d '30 minutes ago' --iso-8601=seconds) \
  --end-time $(date -u --iso-8601=seconds) \
  --period 300 \
  --statistics Average

# 3. Check for traffic spikes
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name RequestCount \
  --dimensions Name=LoadBalancer,Value=app/multimodal-prod-alb/... \
  --start-time $(date -u -d '30 minutes ago' --iso-8601=seconds) \
  --end-time $(date -u --iso-8601=seconds) \
  --period 300 \
  --statistics Sum
```

**Resolution**:

**If CPU/Memory high**:
```bash
# Scale up immediately
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --desired-count 10
```

**If cold starts**:
```bash
# Ensure minimum healthy tasks
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --deployment-configuration "minimumHealthyPercent=100,maximumPercent=200"
```

**If model issue**:
```bash
# Rollback to previous model version (see Section 5)
python scripts/rollback_model.py --version previous
```

**Prevention**:
- Review autoscaling policies
- Add predictive scaling for known traffic patterns
- Implement request queuing

### 4.2 High Error Rate (>1%)

**Symptoms**: Error rate alerts, failed requests

**Diagnosis**:
```bash
# 1. Check error distribution
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "ERROR" \
  --start-time $(date -u -d '1 hour ago' +%s)000 \
  | jq -r '.events[].message' \
  | grep -oP 'error_code":"[^"]*"' \
  | sort | uniq -c | sort -nr

# 2. Check specific error codes
curl https://api.company.com/metrics | grep error_count_total

# 3. Sample failed requests
aws logs tail /aws/ecs/multimodal-prod --filter-pattern "ERROR" --since 10m
```

**Resolution by Error Code**:

**DOC_001 (Corrupt PDF)**:
```bash
# Check if specific file pattern
# Add validation before processing
# Contact client to fix file format
```

**IMG_001 (Corrupt Image)**:
```bash
# Check image preprocessing pipeline
# Verify image upload validation
# Review client integration
```

**SYS_001 (OOM)**:
```bash
# Scale up task memory
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --task-definition multimodal-api:NEW_VERSION  # with higher memory
```

**RATE_001 (Rate Limit)**:
```bash
# Check if legitimate traffic spike
# Adjust rate limits if needed
aws ssm put-parameter \
  --name /multimodal/prod/rate_limit_rpm \
  --value "120" \
  --overwrite

# Or identify and block abusive client
```

**Prevention**:
- Improve input validation
- Add circuit breakers
- Implement better error handling

### 4.3 Model Drift Detected

**Symptoms**: Drift alerts, accuracy degradation, confidence drops

**Diagnosis**:
```bash
# 1. Check drift metrics
curl https://api.company.com/metrics | grep drift_score

# 2. Review drift monitor logs
python -m src.utils.drift_monitor --check --window 1000

# 3. Check recent predictions
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "prediction" \
  --start-time $(date -u -d '24 hours ago' +%s)000 \
  | jq -r '.events[].message' \
  | jq -s 'group_by(.predicted_class) | map({class: .[0].predicted_class, count: length})'
```

**Resolution**:

**If distribution drift**:
```bash
# 1. Collect recent production data
python scripts/export_predictions.py --days 7 --output drift_data.csv

# 2. Trigger retraining pipeline
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:ACCOUNT:stateMachine:model-retraining \
  --input '{"drift_type": "distribution", "data_path": "s3://multimodal-data/drift_data.csv"}'

# 3. Monitor retraining
# Wait for completion notification

# 4. Deploy new model (canary first)
# See Section 3.1
```

**If confidence drift**:
```bash
# 1. Review recent low-confidence predictions
python scripts/analyze_confidence.py --threshold 0.5 --days 3

# 2. Add to training data (if labeled)
python scripts/add_to_training_set.py --source production --days 3

# 3. Schedule retraining
```

**If performance drift** (requires ground truth):
```bash
# 1. Compare with labeled data
python scripts/evaluate_production.py --labeled-file ground_truth.csv

# 2. If accuracy < 85%, trigger immediate retraining
# If accuracy 85-90%, schedule retraining within 48h
# If accuracy > 90%, continue monitoring
```

**Prevention**:
- Continuous monitoring
- Regular retraining schedule (weekly/monthly)
- Maintain diverse training data

### 4.4 Deployment Failure

**Symptoms**: Failed ECS task, health check failures, deployment stuck

**Diagnosis**:
```bash
# 1. Check deployment status
aws ecs describe-services \
  --cluster multimodal-prod \
  --services multimodal-api \
  --query 'services[0].deployments'

# 2. Check task status
aws ecs list-tasks \
  --cluster multimodal-prod \
  --service multimodal-api \
  --desired-status STOPPED

# 3. Get task failure reason
aws ecs describe-tasks \
  --cluster multimodal-prod \
  --tasks TASK_ID \
  --query 'tasks[0].stoppedReason'

# 4. Check logs
aws logs tail /aws/ecs/multimodal-prod --follow
```

**Resolution**:

**If health check failing**:
```bash
# Rollback deployment immediately
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --task-definition multimodal-api:PREVIOUS_VERSION \
  --force-new-deployment

# Check health check configuration
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:...
```

**If model loading failure**:
```bash
# Verify model exists in S3
aws s3 ls s3://multimodal-models-prod/document_classifier/VERSION/

# Check IAM permissions
# Verify model version parameter
aws ssm get-parameter --name /multimodal/prod/document_classifier_version

# Rollback to previous model
python scripts/rollback_model.py --model document_classifier --version previous
```

**If resource issues**:
```bash
# Increase task resources
# Update task definition with higher CPU/memory
# Redeploy
```

**Prevention**:
- Always test in staging first
- Use blue/green deployments
- Implement automated rollback on health check failure

### 4.5 Complete Service Outage

**Symptoms**: All requests failing, 503 errors, no healthy tasks

**Immediate Actions** (< 5 minutes):

```bash
# 1. Check ECS service status
aws ecs describe-services --cluster multimodal-prod --services multimodal-api

# 2. Check ALB target health
aws elbv2 describe-target-health --target-group-arn TARGET_GROUP_ARN

# 3. Emergency scale-up
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --desired-count 10

# 4. If deployment issue, force rollback
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --task-definition multimodal-api:LAST_KNOWN_GOOD_VERSION \
  --force-new-deployment

# 5. Notify stakeholders
# Post in #incidents Slack channel
# Page on-call engineer
```

**Escalation**:
- 0-5 min: On-call ML engineer
- 5-15 min: ML platform lead
- 15+ min: VP Engineering + Customer success

**Post-Incident**:
- Write incident report
- Schedule postmortem
- Implement prevention measures

---

## 5. Rollback Procedures

### 5.1 Rollback Model Version

**Use the model versioning system**:

```bash
# Check current version
python -m src.utils.model_versioning --list

# Output:
# v5 (current): accuracy=0.92, deployed=2025-11-15
# v4: accuracy=0.91, deployed=2025-11-08
# v3: accuracy=0.90, deployed=2025-11-01

# Rollback to previous version
python -m src.utils.model_versioning --rollback 1

# Or rollback to specific version
python -m src.utils.model_versioning --load v4

# Verify rollback
curl https://api.company.com/model-info
```

**Manual rollback** (if script fails):

```bash
# 1. Update model version parameter
aws ssm put-parameter \
  --name /multimodal/prod/document_classifier_version \
  --value "v4" \
  --overwrite

# 2. Restart tasks
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --force-new-deployment

# 3. Monitor health
watch -n 5 'curl -s https://api.company.com/health'
```

### 5.2 Rollback Code Deployment

**Using ECS task definition versions**:

```bash
# 1. List recent task definitions
aws ecs list-task-definitions \
  --family-prefix multimodal-api \
  --max-results 10 \
  --sort DESC

# 2. Rollback to previous version
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --task-definition multimodal-api:PREVIOUS_VERSION

# 3. Monitor deployment
aws ecs wait services-stable \
  --cluster multimodal-prod \
  --services multimodal-api
```

**Using Git tags** (rebuild and redeploy):

```bash
# 1. Find last stable tag
git tag -l --sort=-version:refname | head -5

# 2. Checkout stable version
git checkout v1.4.2

# 3. Rebuild Docker image
docker build -t multimodal-api:v1.4.2 .

# 4. Push to ECR
docker tag multimodal-api:v1.4.2 ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/multimodal-api:v1.4.2
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/multimodal-api:v1.4.2

# 5. Update task definition to use new image
# 6. Deploy
```

### 5.3 Rollback Configuration

```bash
# 1. View parameter history
aws ssm get-parameter-history \
  --name /multimodal/prod/max_batch_size

# 2. Restore previous value
aws ssm put-parameter \
  --name /multimodal/prod/max_batch_size \
  --value "16" \
  --overwrite

# 3. Restart services to apply
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-api \
  --force-new-deployment
```

---

## 6. Monitoring & Alerts

### 6.1 Key Metrics to Watch

**Request Metrics**:
- `inference_requests_total`: Total requests
- `inference_latency_seconds`: Latency distribution
- `error_count_total`: Error counts by type

**Model Metrics**:
- `model_confidence_scores`: Confidence distribution
- `prediction_distribution_total`: Class distribution
- `drift_score`: Drift detection scores

**Resource Metrics**:
- `cpu_utilization_percent`: CPU usage
- `memory_used_bytes`: Memory usage
- `gpu_utilization_percent`: GPU usage (if applicable)

**Business Metrics**:
- Request rate (req/s)
- Success rate (%)
- P50/P95/P99 latency

### 6.2 Alert Configuration

**Critical Alerts** (Page immediately):

```yaml
# P95 latency > 1000ms for 5 minutes
- alert: HighLatency
  expr: histogram_quantile(0.95, inference_latency_seconds_bucket) > 1.0
  for: 5m
  severity: critical

# Error rate > 5% for 2 minutes
- alert: HighErrorRate
  expr: rate(error_count_total[2m]) / rate(inference_requests_total[2m]) > 0.05
  for: 2m
  severity: critical

# Service down
- alert: ServiceDown
  expr: up{job="multimodal-api"} == 0
  for: 1m
  severity: critical
```

**Warning Alerts** (Slack notification):

```yaml
# P95 latency > 600ms for 10 minutes
- alert: ElevatedLatency
  expr: histogram_quantile(0.95, inference_latency_seconds_bucket) > 0.6
  for: 10m
  severity: warning

# Error rate > 1% for 5 minutes
- alert: ElevatedErrorRate
  expr: rate(error_count_total[5m]) / rate(inference_requests_total[5m]) > 0.01
  for: 5m
  severity: warning

# Drift detected
- alert: ModelDrift
  expr: drift_score{metric_type="distribution"} > 0.1
  for: 15m
  severity: warning
```

### 6.3 Dashboards

**Main Dashboard**: https://grafana.company.com/d/multimodal

Panels:
1. Request rate (5min)
2. P50/P95/P99 latency
3. Error rate by type
4. Success rate
5. CPU/Memory utilization
6. Drift scores
7. Model confidence distribution
8. Request distribution by model type

---

## 7. Troubleshooting Guide

### 7.1 Common Issues

| Issue | Symptom | Likely Cause | Solution |
|-------|---------|--------------|----------|
| Slow inference | P95 > 1000ms | High CPU/batch size | Scale up or optimize batch size |
| OOM errors | Task crashes | Large batch/memory leak | Reduce batch size, check for leaks |
| Model load failure | 500 errors on startup | Missing S3 model/permissions | Verify S3 path and IAM role |
| Drift alerts | Warning emails | Data distribution change | Review data, retrain model |
| High error rate | DOC_001 errors | Client sending corrupt files | Add validation, contact client |
| Rate limit exceeded | RATE_001 errors | Traffic spike or abuse | Increase limits or block client |

### 7.2 Debug Commands

```bash
# Get detailed task information
aws ecs describe-tasks --cluster multimodal-prod --tasks TASK_ID

# Stream live logs
aws logs tail /aws/ecs/multimodal-prod --follow --format short

# Test inference locally
python -m src.inference.predictor \
  --model-type document \
  --input "Test document" \
  --debug

# Profile inference performance
python scripts/profile_resources.py \
  --model-type document \
  --batch-sizes 1,4,8,16

# Check model health
python -m src.utils.model_versioning --health-check
```

### 7.3 Log Analysis

**Find errors in last hour**:
```bash
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "ERROR" \
  --start-time $(date -u -d '1 hour ago' +%s)000 \
  | jq -r '.events[].message'
```

**Analyze error distribution**:
```bash
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "{$.level = ERROR}" \
  --start-time $(date -u -d '24 hours ago' +%s)000 \
  | jq -r '.events[].message | fromjson | .error_code' \
  | sort | uniq -c | sort -nr
```

**Track requests by correlation ID**:
```bash
aws logs filter-log-events \
  --log-group-name /aws/ecs/multimodal-prod \
  --filter-pattern "CORRELATION_ID" \
  | jq -r '.events[].message'
```

---

## 8. Contact Information

### On-Call Rotation

| Role | Primary | Backup |
|------|---------|--------|
| ML Engineer | oncall-ml@company.com | backup-ml@company.com |
| Platform Engineer | oncall-platform@company.com | backup-platform@company.com |
| Manager | ml-manager@company.com | |

### Escalation Path

1. **L1**: On-call ML Engineer (respond within 15 min)
2. **L2**: ML Platform Lead (respond within 30 min)
3. **L3**: VP Engineering (respond within 1 hour)

### Communication Channels

- **Incidents**: #incidents (Slack)
- **Alerts**: #alerts-multimodal (Slack)
- **Team**: #ml-platform (Slack)
- **PagerDuty**: https://company.pagerduty.com/services/multimodal

### External Dependencies

| Service | Contact | SLA |
|---------|---------|-----|
| AWS Support | Premium Support | 1 hour response |
| Model Training Pipeline | data-eng@company.com | 4 hours |
| Client Integration | api-support@company.com | 2 hours |

---

## Appendix: Quick Reference

### Health Check
```bash
curl https://api.company.com/health
```

### View Metrics
```bash
curl https://api.company.com/metrics
```

### Scale Service
```bash
aws ecs update-service --cluster multimodal-prod --service multimodal-api --desired-count N
```

### Rollback Model
```bash
python -m src.utils.model_versioning --rollback 1
```

### Rollback Deployment
```bash
aws ecs update-service --cluster multimodal-prod --service multimodal-api --task-definition multimodal-api:PREVIOUS
```

### View Logs
```bash
aws logs tail /aws/ecs/multimodal-prod --follow
```

### Emergency Stop
```bash
aws ecs update-service --cluster multimodal-prod --service multimodal-api --desired-count 0
```

---

**Document Version**: 1.0
**Next Review**: Quarterly (February 2026)
**Changelog**: See `docs/RUNBOOK_CHANGELOG.md`
