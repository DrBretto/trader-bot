#!/bin/bash
#
# Set up AWS Batch infrastructure for monthly CPU-only model training.
#
# Creates: ECR repo, IAM roles, Batch CE/queue/job-def, EventBridge rule.
# All resources scale to zero when idle. No NAT gateway, no Elastic IPs.
#
set -euo pipefail

REGION="${1:-us-east-1}"
S3_BUCKET="${2:-investment-system-data}"

# Resource names
ECR_REPO="investment-system-training"
CE_NAME="investment-training-spot-ce"
JQ_NAME="investment-training-queue"
JD_NAME="investment-training-job"
INSTANCE_ROLE="investment-training-instance-role"
INSTANCE_PROFILE="investment-training-instance-profile"
JOB_ROLE="investment-system-training-role"
SCHEDULE_RULE="investment-system-monthly-training"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
IMAGE_URI="$ECR_URI/$ECR_REPO:latest"

echo "Setting up Batch training infrastructure..."
echo "  Account:  $ACCOUNT_ID"
echo "  Region:   $REGION"
echo "  Bucket:   $S3_BUCKET"
echo ""

# ─── 1. ECR Repository ─────────────────────────────────────────────
echo "1. ECR repository..."
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1; then
    aws ecr create-repository \
        --repository-name "$ECR_REPO" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=false \
        --query 'repository.repositoryUri' --output text
    echo "   Created $ECR_REPO"
else
    echo "   Already exists"
fi

# ECR lifecycle: keep only last 3 images
aws ecr put-lifecycle-policy \
    --repository-name "$ECR_REPO" \
    --region "$REGION" \
    --lifecycle-policy-text '{
        "rules": [{
            "rulePriority": 1,
            "description": "Keep only 3 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 3
            },
            "action": { "type": "expire" }
        }]
    }' >/dev/null
echo "   Lifecycle: keep last 3 images"

# ─── 2. IAM: Instance role (EC2 instances in Batch CE) ──────────────
echo "2. Instance role..."
if ! aws iam get-role --role-name "$INSTANCE_ROLE" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "$INSTANCE_ROLE" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": { "Service": "ec2.amazonaws.com" },
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null
    aws iam attach-role-policy \
        --role-name "$INSTANCE_ROLE" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
    echo "   Created $INSTANCE_ROLE"
else
    echo "   Already exists"
fi

# Instance profile
if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" >/dev/null
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$INSTANCE_PROFILE" \
        --role-name "$INSTANCE_ROLE"
    echo "   Created instance profile"
    echo "   Waiting for propagation..."
    sleep 15
else
    echo "   Instance profile already exists"
fi

# ─── 3. IAM: Job execution role (container gets this) ──────────────
echo "3. Job execution role..."
if ! aws iam get-role --role-name "$JOB_ROLE" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "$JOB_ROLE" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": { "Service": "ecs-tasks.amazonaws.com" },
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null

    # S3 + logs + SNS (same perms as Lambda role, minus Bedrock)
    aws iam put-role-policy \
        --role-name "$JOB_ROLE" \
        --policy-name "training-policy" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket","s3:DeleteObject"],
                    "Resource": ["arn:aws:s3:::'"$S3_BUCKET"'","arn:aws:s3:::'"$S3_BUCKET"'/*"]
                },
                {
                    "Effect": "Allow",
                    "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
                    "Resource": "arn:aws:logs:*:*:*"
                },
                {
                    "Effect": "Allow",
                    "Action": ["sns:Publish"],
                    "Resource": "arn:aws:sns:'"$REGION"':*:investment-system-alerts"
                },
                {
                    "Effect": "Allow",
                    "Action": ["ce:GetCostAndUsage"],
                    "Resource": "*"
                }
            ]
        }'
    echo "   Created $JOB_ROLE"
else
    echo "   Already exists"
fi

# ─── 4. Batch: Ensure service-linked role exists ────────────────────
echo "4. Batch service-linked role..."
aws iam create-service-linked-role --aws-service-name batch.amazonaws.com 2>/dev/null || true
echo "   OK (exists or created)"

# ─── 5. Batch: Compute Environment (EC2 Spot, scales 0→4→0) ────────
echo "5. Batch compute environment..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --region "$REGION" --query 'Vpcs[0].VpcId' --output text)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --region "$REGION" --query 'Subnets[].SubnetId' --output json)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --region "$REGION" --query 'SecurityGroups[0].GroupId' --output text)

INSTANCE_PROFILE_ARN="arn:aws:iam::${ACCOUNT_ID}:instance-profile/$INSTANCE_PROFILE"

CE_EXISTS=$(aws batch describe-compute-environments --compute-environments "$CE_NAME" --region "$REGION" --query 'computeEnvironments[0].computeEnvironmentName' --output text 2>/dev/null || echo "None")

if [ "$CE_EXISTS" = "None" ] || [ "$CE_EXISTS" = "" ]; then
    aws batch create-compute-environment \
        --compute-environment-name "$CE_NAME" \
        --type MANAGED \
        --state ENABLED \
        --compute-resources '{
            "type": "SPOT",
            "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
            "minvCpus": 0,
            "maxvCpus": 4,
            "desiredvCpus": 0,
            "instanceTypes": ["m5.large", "m5.xlarge", "m5a.large", "m5a.xlarge"],
            "subnets": '"$SUBNETS"',
            "securityGroupIds": ["'"$SG_ID"'"],
            "instanceRole": "'"$INSTANCE_PROFILE_ARN"'",
            "spotIamFleetRole": "arn:aws:iam::'"$ACCOUNT_ID"':role/aws-ec2-spot-fleet-tagging-role"
        }' \
        --region "$REGION" >/dev/null
    echo "   Created $CE_NAME (Spot, min=0, max=4 vCPUs)"

    echo "   Waiting for CE to become VALID..."
    for i in $(seq 1 30); do
        STATUS=$(aws batch describe-compute-environments --compute-environments "$CE_NAME" --region "$REGION" --query 'computeEnvironments[0].status' --output text 2>/dev/null)
        if [ "$STATUS" = "VALID" ]; then
            echo "   CE is VALID"
            break
        fi
        sleep 5
    done
else
    echo "   Already exists"
fi

# ─── 6. Batch: Job Queue ────────────────────────────────────────────
echo "6. Batch job queue..."
JQ_EXISTS=$(aws batch describe-job-queues --job-queues "$JQ_NAME" --region "$REGION" --query 'jobQueues[0].jobQueueName' --output text 2>/dev/null || echo "None")

if [ "$JQ_EXISTS" = "None" ] || [ "$JQ_EXISTS" = "" ]; then
    CE_ARN=$(aws batch describe-compute-environments --compute-environments "$CE_NAME" --region "$REGION" --query 'computeEnvironments[0].computeEnvironmentArn' --output text)
    aws batch create-job-queue \
        --job-queue-name "$JQ_NAME" \
        --state ENABLED \
        --priority 1 \
        --compute-environment-order "order=1,computeEnvironment=$CE_ARN" \
        --region "$REGION" >/dev/null
    echo "   Created $JQ_NAME"
else
    echo "   Already exists"
fi

# ─── 7. Build and push training image ──────────────────────────────
echo "7. Building training container image..."
cd "$(dirname "$0")/.."

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"

docker buildx build --platform linux/amd64 --progress=plain \
    -f Dockerfile.training \
    -t "$IMAGE_URI" \
    --provenance=false \
    --push \
    .

IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name "$ECR_REPO" \
    --image-ids imageTag=latest \
    --region "$REGION" \
    --query 'imageDetails[0].imageDigest' --output text)
IMAGE_WITH_DIGEST="$ECR_URI/$ECR_REPO@$IMAGE_DIGEST"
echo "   Pushed: $IMAGE_WITH_DIGEST"

# ─── 8. Batch: Job Definition ──────────────────────────────────────
echo "8. Batch job definition..."
JOB_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/$JOB_ROLE"

aws batch register-job-definition \
    --job-definition-name "$JD_NAME" \
    --type container \
    --container-properties '{
        "image": "'"$IMAGE_WITH_DIGEST"'",
        "vcpus": 4,
        "memory": 14336,
        "jobRoleArn": "'"$JOB_ROLE_ARN"'",
        "environment": [
            {"name": "S3_BUCKET", "value": "'"$S3_BUCKET"'"},
            {"name": "AWS_REGION_NAME", "value": "'"$REGION"'"}
        ],
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": "/aws/batch/investment-training",
                "awslogs-region": "'"$REGION"'",
                "awslogs-stream-prefix": "training"
            }
        }
    }' \
    --timeout '{"attemptDurationSeconds": 14400}' \
    --retry-strategy '{"attempts": 2}' \
    --region "$REGION" >/dev/null
echo "   Registered $JD_NAME (4 vCPU, 14 GB, 4hr timeout, 2 retries)"

# ─── 9. CloudWatch log group with retention ─────────────────────────
echo "9. CloudWatch log group..."
aws logs create-log-group --log-group-name "/aws/batch/investment-training" --region "$REGION" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/batch/investment-training" --retention-in-days 30 --region "$REGION"
echo "   /aws/batch/investment-training (30-day retention)"

# ─── 10. EventBridge: Monthly cron ──────────────────────────────────
echo "10. EventBridge monthly schedule..."

# Create or update rule: 1st of month at 6 AM UTC (2 AM ET)
aws events put-rule \
    --name "$SCHEDULE_RULE" \
    --schedule-expression "cron(0 6 1 * ? *)" \
    --state ENABLED \
    --description "Monthly training: 1st of month at 2 AM ET" \
    --region "$REGION" >/dev/null

# Target = Batch SubmitJob
JD_ARN=$(aws batch describe-job-definitions --job-definition-name "$JD_NAME" --status ACTIVE --region "$REGION" --query 'jobDefinitions[-1].jobDefinitionArn' --output text)
JQ_ARN=$(aws batch describe-job-queues --job-queues "$JQ_NAME" --region "$REGION" --query 'jobQueues[0].jobQueueArn' --output text)

# EventBridge needs a role to submit Batch jobs
EB_ROLE="investment-training-eventbridge-role"
if ! aws iam get-role --role-name "$EB_ROLE" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "$EB_ROLE" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": { "Service": "events.amazonaws.com" },
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null
    aws iam put-role-policy \
        --role-name "$EB_ROLE" \
        --policy-name "submit-batch-job" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": "batch:SubmitJob",
                "Resource": "*"
            }]
        }'
    echo "   Created EventBridge role"
    sleep 10
fi

EB_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/$EB_ROLE"

aws events put-targets \
    --rule "$SCHEDULE_RULE" \
    --targets '[{
        "Id": "monthly-training",
        "Arn": "'"$JQ_ARN"'",
        "RoleArn": "'"$EB_ROLE_ARN"'",
        "BatchParameters": {
            "JobDefinition": "'"$JD_ARN"'",
            "JobName": "monthly-training"
        }
    }]' \
    --region "$REGION" >/dev/null
echo "   Rule: $SCHEDULE_RULE -> $JQ_NAME"

# ─── Done ───────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Batch training infrastructure ready!"
echo ""
echo "  ECR:       $IMAGE_URI"
echo "  CE:        $CE_NAME (Spot, 0-4 vCPUs)"
echo "  Queue:     $JQ_NAME"
echo "  Job Def:   $JD_NAME"
echo "  Schedule:  1st of month at 2 AM ET"
echo ""
echo "To test manually:"
echo "  aws batch submit-job \\"
echo "    --job-name test-training \\"
echo "    --job-queue $JQ_NAME \\"
echo "    --job-definition $JD_NAME \\"
echo "    --region $REGION"
echo ""
echo "To check status:"
echo "  aws batch list-jobs --job-queue $JQ_NAME --region $REGION"
echo "=========================================="
