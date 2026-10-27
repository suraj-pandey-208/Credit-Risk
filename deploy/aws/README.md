# AWS ECS Deployment

Recommended production topology:
- ECR: container image registry
- ECS Fargate: API and Streamlit services
- RDS PostgreSQL: prediction database
- ALB: route `/api/*` to FastAPI and `/` to Streamlit (or use separate hostnames)
- CloudWatch: application logs and alarms
- S3: model/artifact storage if artifacts are externalized

## Build and push

```bash
aws ecr create-repository --repository-name credit-risk-ai
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com
docker build -t credit-risk-ai .
docker tag credit-risk-ai:latest <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/credit-risk-ai:latest
docker push <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/credit-risk-ai:latest
```

Use `task-definition-api.json` and `task-definition-streamlit.json` as starting ECS task definitions. Store DB credentials in AWS Secrets Manager rather than hard-coding them.
