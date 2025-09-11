# Karpenter Organization and Circular Dependency Resolution

## The Circular Dependency Challenge

**The Problem:**
- Karpenter controller (application) needs a running EKS cluster
- NodePools (infrastructure config) need Karpenter controller running  
- But NodePools define infrastructure that applications depend on
- EKS cluster needs some initial nodes to run Karpenter

**The Solution:**
Bootstrap pattern with layered deployment

## Recommended File Organization

### Approach 1: Four-Layer Architecture (Recommended)

```
terraform-eks-platform/
├── infrastructure/
│   ├── networking/
│   │   ├── main.tf                 # VPC, subnets
│   │   ├── outputs.tf
│   │   └── terraform.tf
│   ├── eks-cluster/
│   │   ├── main.tf                 # EKS cluster
│   │   ├── bootstrap-nodes.tf      # Initial managed node group or Fargate
│   │   ├── karpenter-iam.tf        # IAM roles for Karpenter controller & nodes
│   │   ├── outputs.tf              # Cluster info, IAM role ARNs
│   │   └── terraform.tf
├── platform-services/
│   ├── karpenter-controller/
│   │   ├── main.tf                 # Karpenter Helm chart deployment
│   │   ├── data.tf                 # Data sources from EKS cluster
│   │   ├── variables.tf
│   │   └── terraform.tf
├── platform-configuration/
│   ├── karpenter-nodepools/
│   │   ├── main.tf                 # Default NodePools & EC2NodeClass
│   │   ├── nodepool-general.tf     # General-purpose nodes
│   │   ├── nodepool-compute.tf     # CPU-intensive workloads
│   │   ├── nodepool-memory.tf      # Memory-intensive workloads
│   │   ├── data.tf                 # Data sources from EKS cluster
│   │   └── terraform.tf
└── applications/
    ├── app-team-alpha/
    │   ├── main.tf                 # App deployments
    │   ├── nodepool-specific.tf    # App-specific NodePools (if needed)
    │   └── terraform.tf
```

### Karpenter Component Breakdown

#### Layer 1: Infrastructure (`infrastructure/eks-cluster/`)

```hcl
# infrastructure/eks-cluster/bootstrap-nodes.tf
resource "aws_eks_node_group" "bootstrap" {
  count           = var.enable_bootstrap_nodes ? 1 : 0
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-bootstrap"
  node_role_arn   = aws_iam_role.node_group.arn
  subnet_ids      = var.private_subnet_ids

  capacity_type  = "ON_DEMAND"
  instance_types = ["t3.medium"]

  scaling_config {
    desired_size = 2
    max_size     = 4
    min_size     = 1
  }

  tags = {
    "karpenter.sh/discovery" = var.cluster_name
    "Purpose"               = "bootstrap"
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }
}

# infrastructure/eks-cluster/karpenter-iam.tf
# IAM role for Karpenter controller
resource "aws_iam_role" "karpenter_controller" {
  name = "${var.cluster_name}-karpenter-controller"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRoleWithWebIdentity"
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Condition = {
        StringEquals = {
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:karpenter:karpenter"
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

# IAM role for nodes created by Karpenter
resource "aws_iam_role" "karpenter_node_instance_profile" {
  name = "${var.cluster_name}-karpenter-node-instance-profile"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}
```

#### Layer 2: Platform Services (`platform-services/karpenter-controller/`)

```hcl
# platform-services/karpenter-controller/main.tf
resource "helm_release" "karpenter" {
  name       = "karpenter"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = var.karpenter_version
  namespace  = "karpenter"

  create_namespace = true

  values = [
    yamlencode({
      settings = {
        clusterName     = data.aws_eks_cluster.main.name
        clusterEndpoint = data.aws_eks_cluster.main.endpoint
        interruptionQueue = aws_sqs_queue.karpenter.name
      }
      serviceAccount = {
        annotations = {
          "eks.amazonaws.com/role-arn" = data.terraform_remote_state.eks.outputs.karpenter_controller_role_arn
        }
      }
      controller = {
        resources = {
          requests = {
            cpu    = "1"
            memory = "1Gi"
          }
        }
      }
    })
  ]

  depends_on = [
    data.aws_eks_cluster.main
  ]
}

# Create SQS queue for spot interruption handling
resource "aws_sqs_queue" "karpenter" {
  name = "${data.aws_eks_cluster.main.name}-karpenter"
}
```

#### Layer 3: Platform Configuration (`platform-configuration/karpenter-nodepools/`)

```hcl
# platform-configuration/karpenter-nodepools/main.tf

# Default EC2NodeClass
resource "kubernetes_manifest" "default_ec2nodeclass" {
  manifest = {
    apiVersion = "karpenter.k8s.aws/v1beta1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "default"
    }
    spec = {
      instanceStorePolicy = "RAID0"
      amiFamily          = "AL2"
      subnetSelectorTerms = [{
        tags = {
          "karpenter.sh/discovery" = data.aws_eks_cluster.main.name
        }
      }]
      securityGroupSelectorTerms = [{
        tags = {
          "karpenter.sh/discovery" = data.aws_eks_cluster.main.name
        }
      }]
      role = data.terraform_remote_state.eks.outputs.karpenter_node_instance_profile_name
    }
  }

  depends_on = [data.kubernetes_service.karpenter]
}

# General purpose NodePool
resource "kubernetes_manifest" "general_nodepool" {
  manifest = {
    apiVersion = "karpenter.sh/v1beta1"
    kind       = "NodePool"
    metadata = {
      name = "general-purpose"
    }
    spec = {
      template = {
        metadata = {
          labels = {
            "node-type" = "general-purpose"
          }
        }
        spec = {
          nodeClassRef = {
            apiVersion = "karpenter.k8s.aws/v1beta1"
            kind       = "EC2NodeClass"
            name       = "default"
          }
          requirements = [
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["spot", "on-demand"]
            },
            {
              key      = "node.kubernetes.io/instance-type"
              operator = "In"
              values   = ["m5.large", "m5.xlarge", "m5.2xlarge"]
            }
          ]
        }
      }
      limits = {
        cpu = "1000"
      }
      disruption = {
        consolidationPolicy = "WhenUnderutilized"
        consolidateAfter    = "30s"
        expireAfter         = "2160h" # 90 days
      }
    }
  }

  depends_on = [kubernetes_manifest.default_ec2nodeclass]
}
```

## Deployment Sequence

The key to resolving the circular dependency is the **deployment sequence**:

### 1. Bootstrap Phase
```bash
# Deploy infrastructure with bootstrap nodes
cd infrastructure/networking && terraform apply
cd infrastructure/eks-cluster && terraform apply
```

### 2. Platform Services Phase  
```bash
# Deploy Karpenter controller (runs on bootstrap nodes)
cd platform-services/karpenter-controller && terraform apply
```

### 3. Platform Configuration Phase
```bash
# Deploy default NodePools (now Karpenter is running)
cd platform-configuration/karpenter-nodepools && terraform apply
```

### 4. Application Phase
```bash
# Deploy applications (can now use Karpenter nodes)
cd applications/app-team-alpha && terraform apply
```

### 5. Cleanup Phase (Optional)
```bash
# Remove bootstrap nodes once Karpenter is managing nodes
cd infrastructure/eks-cluster
terraform apply -var="enable_bootstrap_nodes=false"
```

## Alternative: Single-Layer with Depends_on

If you prefer a simpler approach, you can keep everything in the platform layer:

```
terraform-eks-platform/
├── infrastructure/
│   ├── eks-cluster/        # EKS + Bootstrap nodes + Karpenter IAM
├── platform-addons/
│   ├── main.tf             # Karpenter controller
│   ├── karpenter-nodepools.tf  # NodePools with explicit depends_on
│   └── terraform.tf
```

```hcl
# platform-addons/karpenter-nodepools.tf
resource "kubernetes_manifest" "general_nodepool" {
  # ... nodepool configuration ...
  
  depends_on = [
    helm_release.karpenter,
    time_sleep.wait_for_karpenter
  ]
}

resource "time_sleep" "wait_for_karpenter" {
  depends_on = [helm_release.karpenter]
  create_duration = "60s"
}
```

## Best Practices

1. **Always use bootstrap nodes** - Don't rely on Karpenter for initial cluster operation
2. **Separate Karpenter controller from NodePools** - Different lifecycles and responsibilities
3. **Use explicit dependencies** - Ensure proper ordering with `depends_on`
4. **Tag resources for discovery** - Use consistent tagging for Karpenter discovery
5. **Monitor deployment sequence** - Deploy layers sequentially, not all at once
6. **Plan for updates** - NodePool changes require careful coordination

## When Each Component Changes

- **Infrastructure layer**: Rarely (cluster upgrades, IAM changes)
- **Controller layer**: Occasionally (Karpenter version updates)  
- **NodePool layer**: Regularly (capacity adjustments, instance type changes)
- **Application layer**: Frequently (deployments, scaling)

This organization prevents the circular dependency while maintaining clear separation of concerns.