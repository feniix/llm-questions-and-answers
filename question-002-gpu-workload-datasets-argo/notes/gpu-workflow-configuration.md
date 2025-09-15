# Argo Workflows GPU Configuration (Assumes Argo is installed)

## GPU Workflow Controller Configuration

```yaml
# workflow-controller-configmap-patch.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-controller-configmap
  namespace: argo
data:
  # Increase resource limits for GPU artifact handling
  executor: |
    resources:
      requests:
        cpu: 200m
        memory: 128Mi
      limits:
        cpu: 1000m
        memory: 2Gi
  # Configure container runtime
  containerRuntimeExecutor: docker
  # Set parallelism limits for GPU workloads
  parallelism: 8
  # Archive workflows with longer TTL for GPU results
  persistence: |
    archive: true
    archiveTTL: 14d
  # GPU-specific workflow defaults
  workflowDefaults: |
    activeDeadlineSeconds: 7200  # 2 hours max for GPU workflows
    ttlStrategy:
      secondsAfterCompletion: 300
      secondsAfterFailure: 300
```

```bash
# Apply the GPU-optimized configuration
kubectl apply -f workflow-controller-configmap-patch.yaml

# Restart workflow controller to pick up changes
kubectl rollout restart deployment/workflow-controller -n argo
```

## Deploy GPU Workflow Templates

```bash
# Deploy all GPU workflow templates
kubectl apply -f gpu-workflow-templates.yaml

# Verify templates are available
argo template list -n argo
```

## GPU Node Verification

### Check GPU Nodes

```bash
# Check GPU nodes are available
kubectl get nodes -l node-type=gpu-t4 -o wide

# Verify GPU resources
kubectl describe nodes -l node-type=gpu-t4 | grep nvidia.com/gpu

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia
```

### Test GPU Access

```bash
# Quick GPU test
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: argo
spec:
  nodeSelector:
    node-type: gpu-t4
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-runtime-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
EOF

# Check results and cleanup
kubectl logs gpu-test -n argo
kubectl delete pod gpu-test -n argo
```

## Submit GPU Workflows

### Embedding Generation

```bash
# Small test (1K documents)
argo submit -n argo --from workflowtemplate/gpu-embedding-generation \
  --parameter batch-size=16 \
  --parameter dataset-size=1000 \
  --name embedding-test-$(date +%s)

# Production scale (100K documents)
argo submit -n argo --from workflowtemplate/gpu-embedding-generation \
  --parameter batch-size=32 \
  --parameter dataset-size=100000 \
  --parameter model-name="sentence-transformers/all-mpnet-base-v2" \
  --name embedding-prod-$(date +%s)
```

### Document Summarization

```bash
# Fast processing
argo submit -n argo --from workflowtemplate/gpu-document-summarization \
  --parameter batch-size=12 \
  --parameter max-length=100 \
  --name summarization-fast-$(date +%s)

# High quality summaries
argo submit -n argo --from workflowtemplate/gpu-document-summarization \
  --parameter batch-size=8 \
  --parameter max-length=200 \
  --parameter min-length=75 \
  --name summarization-quality-$(date +%s)
```

### Hyperparameter Tuning

```bash
# Quick tuning (4 trials)
argo submit -n argo --from workflowtemplate/gpu-hyperparameter-tuning \
  --parameter max-trials=4 \
  --name hyperparam-quick-$(date +%s)

# Comprehensive search (16 trials)
argo submit -n argo --from workflowtemplate/gpu-hyperparameter-tuning \
  --parameter max-trials=16 \
  --name hyperparam-full-$(date +%s)
```

## Monitor GPU Workflows

```bash
# List all workflows
argo list -n argo

# Watch specific workflow
argo watch <workflow-name> -n argo

# Get workflow logs
argo logs <workflow-name> -n argo

# Monitor GPU usage
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

## Performance Expectations (T4 GPUs)

- **Embedding Generation**: ~3K embeddings/minute per T4
- **Document Summarization**: ~10-15 documents/minute per T4
- **Hyperparameter Tuning**: Parallel execution across available T4 nodes
- **Cost**: ~$0.526/hour per g4dn.2xlarge instance

## Cleanup

```bash
# Delete completed workflows
argo delete --completed -n argo

# Delete old workflows (older than 7 days)
argo delete --older 7d -n argo

# Clean up monitoring pod
kubectl delete pod gpu-monitor -n argo
```

Your GPU workflows are now ready to run on T4 instances with optimized performance and cost efficiency!
