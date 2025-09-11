# Terraform File Organization for EKS + Kubernetes Deployments

## Question

How should I organize the file tree for a Terraform configuration that sets up an EKS cluster with Kubernetes application deployments? What are the best practices for managing dependencies between infrastructure and applications - should I use remote state references or data sources? How do I handle complex components like Karpenter that create circular dependencies between infrastructure and application layers?

## Executive Summary

When managing both EKS infrastructure and Kubernetes applications with Terraform, the key is to **separate concerns based on lifecycle, team ownership, and coupling**. The recommended approach uses **layered architecture with separate state files** to reduce blast radius, enable parallel team work, and maintain clear dependency boundaries.

## Key Principles

1. **Separate State Files** - Different components with different lifecycles should have separate state files
2. **Team Boundaries** - Platform teams manage infrastructure, application teams manage their deployments  
3. **Dependency Flow** - Infrastructure layer provides outputs consumed by application layer
4. **Reduced Blast Radius** - Changes to applications don't affect infrastructure and vice versa

## Recommended Approaches

### Approach 1: Layered Architecture (Recommended)

This approach separates infrastructure concerns by lifecycle and team ownership:

```text
terraform-eks-platform/
├── infrastructure/
│   ├── networking/
│   │   ├── main.tf                 # VPC, subnets, NAT gateways
│   │   ├── outputs.tf              # VPC ID, subnet IDs, etc.
│   │   └── terraform.tf            # Remote state config
│   ├── eks-cluster/
│   │   ├── main.tf                 # EKS cluster, node groups, IRSA
│   │   ├── variables.tf            # Cluster version, instance types
│   │   ├── outputs.tf              # Cluster endpoint, OIDC issuer
│   │   ├── data.tf                 # Data sources from networking layer
│   │   └── terraform.tf
│   └── platform-addons/
│       ├── main.tf                 # AWS Load Balancer Controller, EBS CSI
│       ├── helm-releases.tf        # Core platform Helm charts
│       └── terraform.tf
├── applications/
│   ├── app-team-alpha/
│   │   ├── main.tf                 # Kubernetes deployments, services
│   │   ├── kubernetes-resources.tf # ConfigMaps, Secrets, Ingress
│   │   ├── data.tf                 # Data sources from EKS cluster
│   │   └── terraform.tf
│   └── app-team-beta/
│       └── [similar structure]
├── modules/
│   ├── eks-cluster/               # Reusable EKS module
│   ├── application-stack/         # Standard app deployment pattern
│   └── monitoring/                # Observability stack
└── environments/
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars
```

**Benefits:**

- **Separate state files** for each layer (reduces blast radius)
- **Clear dependencies** flow from infrastructure → applications
- **Team boundaries** respected (platform team owns infrastructure)
- **Independent deployments** for applications
- **Scalable** - easy to add new applications or infrastructure components

**When to use:** Medium to large organizations with dedicated platform teams

### Approach 2: Team-Based Organization

Organize by team ownership and operational boundaries:

```text
terraform-platform/
├── platform-team/
│   ├── shared-infrastructure/
│   │   ├── networking/
│   │   ├── eks-clusters/
│   │   └── shared-services/         # Monitoring, logging, security
│   └── environments/
│       ├── dev/
│       ├── staging/
│       └── prod/
├── application-teams/
│   ├── team-alpha/
│   │   ├── applications/
│   │   │   ├── web-api/
│   │   │   └── worker-service/
│   │   └── environments/
│   └── team-beta/
│       └── [similar structure]
└── shared-modules/
    ├── k8s-application/
    ├── database/
    └── monitoring/
```

**When to use:** Large organizations with strong team autonomy and clear ownership boundaries

### Approach 3: Environment-First (Simple)

For smaller teams or simpler deployments:

```text
terraform-eks/
├── environments/
│   ├── dev/
│   │   ├── infrastructure/
│   │   │   ├── networking.tf
│   │   │   ├── eks-cluster.tf
│   │   │   └── platform-addons.tf
│   │   └── applications/
│   │       ├── app-deployments.tf
│   │       └── app-services.tf
│   ├── staging/
│   └── prod/
└── modules/
    ├── eks/
    └── k8s-app/
```

**When to use:** Small teams, POC environments, or when organizational complexity is low

## State File Strategy

### Recommended State Separation

```text
# Infrastructure states (managed by platform team)
eks-platform-networking-{env}.tfstate
eks-platform-cluster-{env}.tfstate
eks-platform-addons-{env}.tfstate

# Application states (managed by app teams)
app-team-alpha-{env}.tfstate
app-team-beta-{env}.tfstate
```

### Remote State Configuration

```hcl
# infrastructure/eks-cluster/terraform.tf
terraform {
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "eks-platform/cluster/prod/terraform.tfstate"
    region = "us-west-2"
    
    # Enable state locking
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

# applications/app-team-alpha/terraform.tf
terraform {
  backend "s3" {
    bucket = "your-terraform-state-bucket" 
    key    = "applications/team-alpha/prod/terraform.tfstate"
    region = "us-west-2"
    
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}
```

## Dependency Management

You have two main approaches for managing dependencies between your infrastructure and application layers:

### Approach 1: Remote State References (Explicit Dependencies)

This creates explicit dependencies through remote state outputs:

```hcl
# applications/app-team-alpha/data.tf
data "terraform_remote_state" "eks_cluster" {
  backend = "s3"
  config = {
    bucket = "your-terraform-state-bucket"
    key    = "eks-platform/cluster/prod/terraform.tfstate" 
    region = "us-west-2"
  }
}

data "terraform_remote_state" "networking" {
  backend = "s3"
  config = {
    bucket = "your-terraform-state-bucket"
    key    = "eks-platform/networking/prod/terraform.tfstate"
    region = "us-west-2"
  }
}

# applications/app-team-alpha/main.tf
resource "kubernetes_deployment" "app" {
  # Use data from EKS cluster state
  depends_on = [data.terraform_remote_state.eks_cluster]
  
  metadata {
    namespace = kubernetes_namespace.app_namespace.metadata[0].name
  }
  
  spec {
    template {
      spec {
        service_account_name = kubernetes_service_account.app_sa.metadata[0].name
        # ... rest of deployment config
      }
    }
  }
}

# Reference cluster endpoint and CA certificate from infrastructure layer
locals {
  cluster_endpoint = data.terraform_remote_state.eks_cluster.outputs.cluster_endpoint
  cluster_ca_cert  = data.terraform_remote_state.eks_cluster.outputs.cluster_certificate_authority_data
}
```

### Approach 2: Data Sources (Loose Coupling - Recommended for many scenarios)

This approach discovers infrastructure resources directly without referencing remote state:

```hcl
# applications/app-team-alpha/data.tf

# Discover EKS cluster by name or tags
data "aws_eks_cluster" "main" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "main" {
  name = var.cluster_name
}

# Discover VPC and subnets by tags
data "aws_vpc" "main" {
  tags = {
    Name        = "eks-platform-vpc"
    Environment = var.environment
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  
  tags = {
    Type = "private"
  }
}

# Discover security groups
data "aws_security_groups" "eks_additional" {
  tags = {
    "eks-cluster" = var.cluster_name
    "Type"        = "additional"
  }
}

# Discover IAM role for IRSA
data "aws_iam_role" "app_role" {
  name = "${var.cluster_name}-${var.app_name}-irsa-role"
}

# Discover existing Kubernetes resources
data "kubernetes_namespace" "system" {
  count = var.create_namespace ? 0 : 1
  metadata {
    name = var.namespace
  }
}

# applications/app-team-alpha/main.tf
resource "kubernetes_deployment" "app" {
  metadata {
    namespace = var.create_namespace ? kubernetes_namespace.app[0].metadata[0].name : var.namespace
  }
  
  spec {
    template {
      spec {
        service_account_name = kubernetes_service_account.app_sa.metadata[0].name
        
        # Reference discovered security groups
        security_context {
          fs_group = 2000
        }
        
        # ... rest of deployment config
      }
    }
  }
}

# Use discovered cluster information
locals {
  cluster_endpoint = data.aws_eks_cluster.main.endpoint
  cluster_ca_cert  = data.aws_eks_cluster.main.certificate_authority[0].data
  
  # Use discovered VPC and subnets for load balancer annotations
  subnet_ids = data.aws_subnets.private.ids
  vpc_id     = data.aws_vpc.main.id
}

# Configure Kubernetes provider using discovered cluster
provider "kubernetes" {
  host                   = data.aws_eks_cluster.main.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.main.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.main.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.main.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.main.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.main.token
  }
}
```

### Resource Tagging Strategy for Data Source Discovery

For the data source approach to work effectively, implement consistent tagging:

```hcl
# infrastructure/networking/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name         = "${var.cluster_name}-vpc"
    Environment  = var.environment
    ManagedBy    = "terraform"
    Layer        = "networking"
    Project      = var.project_name
  }
}

resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name                              = "${var.cluster_name}-private-${count.index + 1}"
    Environment                       = var.environment
    Type                             = "private"
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
  }
}

# infrastructure/eks-cluster/main.tf
resource "aws_iam_role" "app_irsa_roles" {
  for_each = var.irsa_roles
  
  name = "${var.cluster_name}-${each.key}-irsa-role"
  
  tags = {
    Environment = var.environment
    Application = each.key
    Type        = "irsa-role"
    Cluster     = var.cluster_name
  }
}
```

### Comparison: Remote State vs Data Sources

| Aspect | Remote State | Data Sources |
|--------|-------------|-------------|
| **Coupling** | Explicit, tight coupling | Implicit, loose coupling |
| **Performance** | Fast (no API calls during plan) | Slower (API calls on each run) |
| **Resilience** | Depends on state backend availability | Works with any resource source |
| **Dependencies** | Must coordinate state backend access | Independent of other Terraform runs |
| **Discoverability** | Requires knowledge of state locations | Relies on resource naming/tagging |
| **Multi-tool compatibility** | Terraform-specific | Works with any infrastructure tool |
| **Error handling** | Fails if state unavailable | Fails if resources don't exist |

### When to Choose Each Approach

**Choose Remote State when:**

- Teams work closely together and coordinate deployments
- You want explicit dependency tracking  
- Performance is critical (frequent planning)
- All infrastructure is managed by Terraform
- You have well-established state management practices

**Choose Data Sources when:**

- Teams prefer loose coupling and independence
- Infrastructure might be created by different tools (CDK, CloudFormation, console)
- You want applications to be resilient to infrastructure refactoring
- New teams join frequently and need simple discovery patterns
- You have good resource tagging practices

### Hybrid Approach

You can combine both approaches where it makes sense:

```hcl
# Use data sources for infrastructure discovery
data "aws_eks_cluster" "main" {
  name = var.cluster_name
}

# Use remote state for tightly coupled components
data "terraform_remote_state" "shared_services" {
  backend = "s3"
  config = {
    bucket = "terraform-state-bucket"
    key    = "shared-services/${var.environment}/terraform.tfstate"
    region = var.aws_region
  }
}
```

## Best Practices and Considerations

### 1. State Management

- **Start with separate states** - Don't create a monolithic state file
- **Use remote backends** with state locking (S3 + DynamoDB)
- **Enable state encryption** for security
- **Plan for state splitting** early in the project lifecycle

### 2. Team Collaboration

- **Implement proper RBAC** - Different teams should only access their state files
- **Use consistent naming** conventions across all components
- **Document dependencies** clearly between layers
- **Establish deployment patterns** and approval processes

### 3. Module Strategy

- **Version your modules** - Pin module versions for stability
- **Create reusable patterns** for common application deployments
- **Maintain module documentation** and examples
- **Test modules independently** before using in production

### 4. CI/CD Integration

- **Separate pipelines** for infrastructure vs applications
- **Implement proper approval gates** for infrastructure changes
- **Use workspace isolation** for different environments
- **Plan for GitOps** - Structure supports ArgoCD/Flux integration

### 5. Security Considerations

- **Use IRSA** (IAM Roles for Service Accounts) for pod-level permissions
- **Implement least privilege** access for Terraform service accounts
- **Encrypt state files** and use secure state backends
- **Audit access** to sensitive infrastructure components

## Anti-Patterns to Avoid

### ❌ Monolithic State File

```text
# DON'T DO THIS - Everything in one state
terraform-monolith/
├── main.tf                    # EKS + all apps in one file
├── variables.tf               # 200+ variables 
└── terraform.tf               # One massive state
```

**Problems:**

- Long deployment times
- High blast radius for changes
- Team collaboration conflicts
- Difficult to maintain and debug

### ❌ Tight Coupling Between Layers

```hcl
# DON'T DO THIS - Direct resource references across boundaries
resource "kubernetes_deployment" "app" {
  # Directly referencing infrastructure resources
  spec {
    template {
      spec {
        node_selector = {
          "kubernetes.io/instance-type" = aws_eks_node_group.main.instance_types[0]
        }
      }
    }
  }
}
```

**Use data sources instead:**

```hcl
data "terraform_remote_state" "eks" {
  # Reference through remote state outputs
}
```

## Migration Strategy

If you already have a monolithic Terraform configuration:

### Phase 1: Prepare New Structure

1. Create new directory structure following layered approach
2. Copy existing code into new structure (don't modify original yet)
3. Split variables and outputs appropriately
4. Test new structure in development environment

### Phase 2: Split State Files

1. Create new backend configurations for each layer
2. Use `terraform state mv` to move resources between state files
3. Update data source references
4. Validate all resources are tracked correctly

### Phase 3: Clean Up

1. Remove unused resources from original configuration
2. Update CI/CD pipelines for new structure
3. Update team access controls
4. Document new processes and dependencies

## Handling Complex Components: Karpenter Circular Dependencies

One of the most challenging aspects of organizing EKS + Kubernetes Terraform configurations is handling components like **Karpenter** that create circular dependencies between infrastructure and application layers.

### The Circular Dependency Challenge

**The Problem:**

- Karpenter controller (application) needs a running EKS cluster
- NodePools (infrastructure config) need Karpenter controller running  
- But NodePools define infrastructure that applications depend on
- EKS cluster needs some initial nodes to run Karpenter

### Solution: Four-Layer Architecture with Bootstrap Pattern

```text
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

### Deployment Sequence

The key to resolving the circular dependency is the **deployment sequence**:

1. **Bootstrap Phase**: Deploy EKS with initial managed node group
2. **Platform Services Phase**: Deploy Karpenter controller (runs on bootstrap nodes)
3. **Platform Configuration Phase**: Deploy NodePools (now Karpenter is available)
4. **Application Phase**: Deploy apps (can now use Karpenter-managed nodes)
5. **Cleanup Phase** (Optional): Remove bootstrap nodes

```bash
# 1. Bootstrap Phase
cd infrastructure/networking && terraform apply
cd infrastructure/eks-cluster && terraform apply

# 2. Platform Services Phase  
cd platform-services/karpenter-controller && terraform apply

# 3. Platform Configuration Phase
cd platform-configuration/karpenter-nodepools && terraform apply

# 4. Application Phase
cd applications/app-team-alpha && terraform apply

# 5. Cleanup Phase (Optional)
cd infrastructure/eks-cluster
terraform apply -var="enable_bootstrap_nodes=false"
```

### Key Insights for Karpenter Organization

- **Bootstrap nodes are essential** - Don't rely on Karpenter for initial cluster operation
- **NodePools are infrastructure config deployed as K8s resources** - They belong in platform configuration layer
- **Separate Karpenter controller from NodePools** - Different lifecycles and responsibilities
- **Use explicit dependencies** - Ensure proper ordering with `depends_on` and `time_sleep` resources

## Conclusion

The layered architecture approach provides the best balance of:

- **Separation of concerns** between infrastructure and applications
- **Team autonomy** with clear ownership boundaries  
- **Scalability** for growing organizations and applications
- **Risk management** through separate state files and blast radius reduction
- **Complex component handling** through bootstrap patterns and sequential deployment

Start with this approach and adapt based on your specific organizational needs and scale.

## References

- [AWS EKS Blueprints Architecture Evolution](https://github.com/aws-ia/terraform-aws-eks-blueprints/blob/main/docs/v4-to-v5/motivation.md)
- [Terraform State Management Best Practices](https://spacelift.io/blog/terraform-state)
- [Splitting Terraform State Files](https://medium.com/@adrianarba/terraform-how-i-split-my-monolithic-state-490916343dba)
- [EKS Blueprints Examples](https://github.com/aws-ia/terraform-aws-eks-blueprints)
