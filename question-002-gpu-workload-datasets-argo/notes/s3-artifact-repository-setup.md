# S3 Artifact Repository Setup with EKS Pod Identity

## Overview

This guide sets up S3 as the artifact repository for Argo Workflows using EKS Pod Identity for secure access without hardcoded credentials.

## Prerequisites

- EKS cluster with Pod Identity Agent installed
- AWS CLI configured with appropriate permissions

## Step 1: Create S3 Bucket

```bash
# Set environment variables
export CLUSTER_NAME="your-eks-cluster-name"
export AWS_REGION="us-west-2"  # Change to your region
export BUCKET_NAME="argo-gpu-artifacts-$(date +%s)"
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create S3 bucket
aws s3 mb s3://${BUCKET_NAME} --region ${AWS_REGION}

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_NAME} \
    --versioning-configuration Status=Enabled

echo "Created bucket: ${BUCKET_NAME}"
```

## Step 2: Create IAM Policy for S3 Access

```bash
# Create IAM policy document
cat > s3-argo-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ArgoArtifactAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
        },
        {
            "Sid": "ArgoBucketAccess",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": "arn:aws:s3:::${BUCKET_NAME}"
        }
    ]
}
EOF

# Create the policy
aws iam create-policy \
    --policy-name ArgoWorkflowsS3Policy \
    --policy-document file://s3-argo-policy.json \
    --description "Policy for Argo Workflows to access S3 artifacts"

# Store policy ARN
export POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/ArgoWorkflowsS3Policy"
echo "Created policy: ${POLICY_ARN}"
```

## Step 3: Create IAM Role for EKS Pod Identity

```bash
# Create trust policy for EKS Pod Identity
cat > trust-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "pods.eks.amazonaws.com"
            },
            "Action": [
                "sts:AssumeRole",
                "sts:TagSession"
            ]
        }
    ]
}
EOF

# Create IAM role for Pod Identity
aws iam create-role \
    --role-name ArgoWorkflowsS3Role \
    --assume-role-policy-document file://trust-policy.json \
    --description "Role for Argo Workflows to access S3 via Pod Identity"

# Attach the S3 policy to the role
aws iam attach-role-policy \
    --role-name ArgoWorkflowsS3Role \
    --policy-arn ${POLICY_ARN}

# Store role ARN
export ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/ArgoWorkflowsS3Role"
echo "Created role: ${ROLE_ARN}"
```

## Step 4: Create Service Account and Pod Identity Association

```bash
# Create Kubernetes service account
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: argo-workflows-sa
  namespace: argo
EOF

# Create Pod Identity Association
aws eks create-pod-identity-association \
    --cluster-name ${CLUSTER_NAME} \
    --namespace argo \
    --service-account argo-workflows-sa \
    --role-arn ${ROLE_ARN}

# Verify the association was created
aws eks list-pod-identity-associations --cluster-name ${CLUSTER_NAME}
```

## Step 4: Configure Artifact Repository

```bash
# Create artifact repository ConfigMap
cat <<EOF | kubectl apply -n argo -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: artifact-repositories
  annotations:
    workflows.argoproj.io/default-artifact-repository: default-s3-artifact-repository
data:
  default-s3-artifact-repository: |
    s3:
      bucket: ${BUCKET_NAME}
      region: ${AWS_REGION}
      endpoint: s3.amazonaws.com
      useSDKCreds: true
      keyFormat: "artifacts/{{workflow.namespace}}/{{workflow.name}}/{{workflow.creationTimestamp}}/{{pod.name}}"
      insecure: false
EOF
```

## Step 5: Update Workflow Controller Configuration

```bash
# Update workflow controller to use the artifact repository
kubectl patch configmap workflow-controller-configmap -n argo --patch '
data:
  artifactRepository: |
    archiveLogs: true
    s3:
      bucket: '${BUCKET_NAME}'
      region: '${AWS_REGION}'
      endpoint: s3.amazonaws.com
      useSDKCreds: true
      keyFormat: "archived-workflows/{{workflow.namespace}}/{{workflow.name}}/{{workflow.uid}}"
'

# Restart workflow controller to pick up changes
kubectl rollout restart deployment/workflow-controller -n argo
```

## Step 6: Test Artifact Repository

```bash
# Create a test workflow with artifacts
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: artifact-test-
  namespace: argo
spec:
  serviceAccountName: argo-workflows-sa
  entrypoint: artifact-example
  templates:
  - name: artifact-example
    dag:
      tasks:
      - name: generate-artifact
        template: generate
      - name: consume-artifact
        template: consume
        dependencies: [generate-artifact]
        arguments:
          artifacts:
          - name: input-file
            from: "{{tasks.generate-artifact.outputs.artifacts.result}}"

  - name: generate
    script:
      image: python:3.9-slim
      command: [python]
      source: |
        import json
        import os

        data = {"message": "Hello from GPU workflow!", "items": list(range(100))}

        os.makedirs('/tmp/outputs', exist_ok=True)
        with open('/tmp/outputs/result.json', 'w') as f:
            json.dump(data, f)

        print("Generated artifact with {} items".format(len(data['items'])))
    outputs:
      artifacts:
      - name: result
        path: /tmp/outputs/result.json

  - name: consume
    inputs:
      artifacts:
      - name: input-file
        path: /tmp/input/result.json
    script:
      image: python:3.9-slim
      command: [python]
      source: |
        import json

        with open('/tmp/input/result.json', 'r') as f:
            data = json.load(f)

        print("Consumed artifact:")
        print(f"Message: {data['message']}")
        print(f"Items count: {len(data['items'])}")
EOF

# Monitor the workflow
argo list -n argo
argo logs -f $(argo list -n argo -o name | head -1) -n argo
```

## Step 7: Verify S3 Storage

```bash
# Check that artifacts were stored in S3
aws s3 ls s3://${BUCKET_NAME}/artifacts/ --recursive

# View artifact content (optional)
# aws s3 cp s3://${BUCKET_NAME}/artifacts/... /tmp/test-artifact.json
# cat /tmp/test-artifact.json
```

## Configuration for Large Artifacts (GPU Models/Datasets)

For large files like ML models or datasets, configure increased limits:

```bash
# Update workflow controller for large artifacts
kubectl patch configmap workflow-controller-configmap -n argo --patch '
data:
  executor: |
    resources:
      requests:
        cpu: 200m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 2Gi
  # Increase archive timeout for large files
  archiveLocation: |
    archiveLogs: true
    s3:
      bucket: '${BUCKET_NAME}'
      region: '${AWS_REGION}'
      endpoint: s3.amazonaws.com
      useSDKCreds: true
  # Set larger timeouts
  workflowDefaults: |
    activeDeadlineSeconds: 3600
'
```

## Troubleshooting

### Check Pod Identity Configuration

```bash
# Verify Pod Identity association
aws eks describe-pod-identity-association \
  --cluster-name ${CLUSTER_NAME} \
  --association-id $(aws eks list-pod-identity-associations \
    --cluster-name ${CLUSTER_NAME} \
    --query 'associations[0].associationId' --output text)

# Check pod environment for AWS credentials
kubectl run debug-pod-identity --rm -i --tty --image=amazon/aws-cli \
  --serviceaccount=argo-workflows-sa -n argo \
  -- aws sts get-caller-identity
```

### Check S3 Access

```bash
# Test S3 access from a pod
kubectl run s3-test --rm -i --tty --image=amazon/aws-cli \
  --serviceaccount=argo-workflows-sa -n argo \
  -- aws s3 ls s3://${BUCKET_NAME}/
```

### Common Issues

1. **Permission denied**: Verify IAM policy has correct bucket ARN
2. **Service account not found**: Ensure IRSA was created in the `argo` namespace
3. **Artifacts not uploading**: Check workflow controller logs for errors

```bash
# Check workflow controller logs
kubectl logs -f deployment/workflow-controller -n argo
```

Your S3 artifact repository is now configured and ready for GPU workflows!

## Environment Variables for Future Use

```bash
# Save these for future reference
echo "export BUCKET_NAME=${BUCKET_NAME}" >> ~/.bashrc
echo "export POLICY_ARN=${POLICY_ARN}" >> ~/.bashrc
source ~/.bashrc
```
