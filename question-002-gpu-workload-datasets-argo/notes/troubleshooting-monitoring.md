# Troubleshooting and Monitoring GPU Workflows

## Quick Deployment Commands

```bash
# Deploy all workflow templates
kubectl apply -f notes/gpu-workflow-templates.yaml

# Submit embedding generation workflow
argo submit -n argo --from workflowtemplate/gpu-embedding-generation \
  --parameter batch-size=32 \
  --parameter dataset-size=5000 \
  --name embedding-test-$(date +%s)

# Submit summarization workflow
argo submit -n argo --from workflowtemplate/gpu-document-summarization \
  --parameter batch-size=8 \
  --name summarization-test-$(date +%s)

# Submit hyperparameter tuning
argo submit -n argo --from workflowtemplate/gpu-hyperparameter-tuning \
  --parameter max-trials=4 \
  --name hyperparam-test-$(date +%s)
```

## Monitoring Commands

```bash
# List all workflows
argo list -n argo

# Watch specific workflow
argo watch <workflow-name> -n argo

# Get workflow logs
argo logs <workflow-name> -n argo

# Get GPU node status
kubectl describe nodes -l node-type=gpu-t4 | grep -A5 "Allocated resources"

# Monitor GPU usage (create monitoring pod first)
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-monitor
  namespace: argo
spec:
  nodeSelector:
    node-type: gpu-t4
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: monitor
    image: nvidia/cuda:11.8-runtime-ubuntu20.04
    command: ["bash", "-c", "while true; do nvidia-smi; echo '---'; sleep 30; done"]
    resources:
      requests:
        nvidia.com/gpu: 1
EOF

# Watch GPU usage
kubectl logs -f gpu-monitor -n argo
```

## Common Issues

### 1. Pod Stuck in Pending

```bash
# Check node resources and taints
kubectl describe node <node-name>
kubectl get pods -n argo -o wide
kubectl describe pod <stuck-pod> -n argo
```

### 2. GPU Not Available

```bash
# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpu: .status.allocatable."nvidia.com/gpu"}'
```

### 3. S3 Access Issues

```bash
# Test Pod Identity
kubectl run s3-test --rm -i --tty --image=amazon/aws-cli \
  --serviceaccount=argo-workflows-sa -n argo \
  -- aws s3 ls s3://$BUCKET_NAME/

# Check Pod Identity association
aws eks describe-pod-identity-association --cluster-name $CLUSTER_NAME \
  --association-id $(aws eks list-pod-identity-associations --cluster-name $CLUSTER_NAME --query 'associations[0].associationId' --output text)
```

### 4. Out of Memory Errors

```bash
# Check memory usage
kubectl top nodes
kubectl top pods -n argo

# Adjust workflow resource limits in templates
# For T4 (16GB VRAM): use batch-size=16-32, memory limits=12-14Gi
```

## Performance Optimization

### Batch Size Guidelines for T4

- **Embedding Generation**: 32-48 documents
- **Summarization**: 8-16 documents
- **Training**: 16-24 samples

### Resource Requests/Limits for T4

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: 8Gi
    cpu: 4
  limits:
    nvidia.com/gpu: 1
    memory: 14Gi  # Leave 2GB for system
    cpu: 6
```

## Workflow Template Usage Examples

### 1. Embedding Generation

```bash
# Small test
argo submit -n argo --from workflowtemplate/gpu-embedding-generation \
  --parameter batch-size=16 --parameter dataset-size=1000

# Production scale
argo submit -n argo --from workflowtemplate/gpu-embedding-generation \
  --parameter batch-size=32 --parameter dataset-size=100000 \
  --parameter model-name="sentence-transformers/all-mpnet-base-v2"
```

### 2. Document Summarization

```bash
# Fast processing
argo submit -n argo --from workflowtemplate/gpu-document-summarization \
  --parameter batch-size=12 --parameter max-length=100

# High quality summaries
argo submit -n argo --from workflowtemplate/gpu-document-summarization \
  --parameter batch-size=8 --parameter max-length=200 --parameter min-length=75
```

### 3. Hyperparameter Tuning

```bash
# Quick tuning
argo submit -n argo --from workflowtemplate/gpu-hyperparameter-tuning \
  --parameter max-trials=4

# Comprehensive search
argo submit -n argo --from workflowtemplate/gpu-hyperparameter-tuning \
  --parameter max-trials=16
```

## Cost Monitoring

```bash
# Check node utilization
kubectl get nodes -o json | jq '.items[] | {
  name: .metadata.name,
  instance: .metadata.labels."node.kubernetes.io/instance-type",
  gpu: .status.allocatable."nvidia.com/gpu"
}'

# Estimated costs (g4dn.2xlarge ~$0.75/hour)
echo "Current GPU nodes: $(kubectl get nodes -l node-type=gpu-t4 --no-headers | wc -l)"
echo "Estimated hourly cost: $$(( $(kubectl get nodes -l node-type=gpu-t4 --no-headers | wc -l) * 75 / 100 ))"
```

## Cleanup

```bash
# Delete completed workflows
argo delete --completed -n argo

# Delete old workflows (older than 7 days)
argo delete --older 7d -n argo

# Clean up monitoring pod
kubectl delete pod gpu-monitor -n argo
```
