# Architecture Diagrams: GPU Workload on Argo Workflows

## System Overview

```mermaid
graph TB
    subgraph "Argo Workflows Cluster"
        direction TB
        CN[Control Node<br/>Argo Server & Controller]

        subgraph "GPU Node 1"
            GP1[GPU Pod<br/>1x T4 GPU<br/>16GB VRAM]
        end

        subgraph "GPU Node 2"
            GP2[GPU Pod<br/>1x T4 GPU<br/>16GB VRAM]
        end

        subgraph "Shared Storage (PVC)"
            direction LR
            INPUT[Input Data<br/>500GB]
            MODELS[Models Cache<br/>50GB]
            RESULTS[Results<br/>Embeddings<br/>100GB]
        end

        CN --> GP1
        CN --> GP2
        GP1 --> INPUT
        GP1 --> MODELS
        GP1 --> RESULTS
        GP2 --> INPUT
        GP2 --> MODELS
        GP2 --> RESULTS
    end

    subgraph "External Sources"
        HF[Hugging Face<br/>Model Hub]
        RP[RedPajama<br/>Dataset]
    end

    HF --> MODELS
    RP --> INPUT
```

## Data Flow Architecture

```mermaid
flowchart TD
    DS[Data Source<br/>RedPajama Dataset<br/>100M+ documents] --> DSP[Data Splitter<br/>CPU Worker]
    DSP --> CQ[Chunk Queue<br/>10K docs per chunk]

    CQ --> GW1[GPU Worker 1<br/>Node 1]
    CQ --> GW2[GPU Worker 2<br/>Node 2]

    subgraph "GPU Worker 1"
        ST1[Sentence Transformer<br/>1x T4 GPU<br/>all-mpnet-base-v2]
        ST1 --> E1[Embeddings<br/>Chunk 1<br/>768-dim vectors]
    end

    subgraph "GPU Worker 2"
        ST2[Sentence Transformer<br/>1x T4 GPU<br/>all-mpnet-base-v2]
        ST2 --> E2[Embeddings<br/>Chunk 2<br/>768-dim vectors]
    end

    GW1 --> E1
    GW2 --> E2

    E1 --> RM[Result Merger<br/>CPU Worker]
    E2 --> RM

    RM --> VDB[Final Vector Database<br/>Indexed Embeddings<br/>Ready for Search]

    style DS fill:#e1f5fe
    style GW1 fill:#f3e5f5
    style GW2 fill:#f3e5f5
    style VDB fill:#e8f5e8
```

## Argo Workflow DAG Structure

```mermaid
flowchart TD
    PD[prepare-data<br/>• Load dataset<br/>• Create splits<br/>• Validate data]

    PD --> GPU1[gpu-embedder-1<br/>• Chunk 1<br/>• 1x T4 GPU<br/>• Batch=32]
    PD --> GPU2[gpu-embedder-2<br/>• Chunk 2<br/>• 1x T4 GPU<br/>• Batch=32]
    PD --> GPU3[gpu-embedder-3<br/>• Chunk 3<br/>• 1x T4 GPU<br/>• Batch=32]
    PD --> GPUN[gpu-embedder-N<br/>• Chunk N<br/>• 1x T4 GPU<br/>• Batch=32]

    GPU1 --> MR[merge-results<br/>• Combine files<br/>• Create index<br/>• Validate output]
    GPU2 --> MR
    GPU3 --> MR
    GPUN --> MR

    style PD fill:#e3f2fd
    style GPU1 fill:#fce4ec
    style GPU2 fill:#fce4ec
    style GPU3 fill:#fce4ec
    style GPUN fill:#fce4ec
    style MR fill:#e8f5e8
```

## GPU Resource Allocation

```mermaid
flowchart TB
    subgraph "Kubernetes Node"
        subgraph "GPU Pod"
            direction LR

            subgraph "GPU:0"
                G0[T4<br/>16GB VRAM]
                P1[Transformer<br/>Process 1<br/>Batch 1-32]
            end

            G0 --> P1
        end

        subgraph "Memory Usage"
            M1[Model: ~3GB per GPU]
            M2[Batch Processing: ~6GB per GPU]
            M3[Available: ~7GB per GPU]
        end

        subgraph "System Resources"
            CPU[16 CPU cores<br/>4 reserved for GPU workers]
            RAM[64GB RAM<br/>32GB for data buffering]
            SSD[NVMe SSD<br/>Fast I/O]
        end
    end

    style G0 fill:#ffebee
    style P1 fill:#e8f5e8
```

## Data Processing Pipeline

```mermaid
flowchart TD
    subgraph "Input"
        DS[RedPajama Dataset<br/>100M+ documents]
    end

    subgraph "Step 1: Data Chunking"
        DC1[Load streaming dataset]
        DC2[Split into 10K document chunks]
        DC3[Serialize chunks to storage]
        DC1 --> DC2 --> DC3
    end

    subgraph "Step 2: Parallel GPU Processing"
        W1[Worker 1<br/>Chunks 1-25<br/>GPU Node 1]
        W2[Worker 2<br/>Chunks 26-50<br/>GPU Node 1]
        W3[Worker 3<br/>Chunks 51-75<br/>GPU Node 2]
        W4[Worker 4<br/>Chunks 76-100<br/>GPU Node 2]
    end

    subgraph "Step 3: Embedding Generation"
        EG1[Model Loading]
        EG2[Batch Processing<br/>32 docs per batch]
        EG3[GPU Memory Management]
        EG4[Output: 768-dim embeddings]
        EG1 --> EG2 --> EG3 --> EG4
    end

    subgraph "Step 4: Result Aggregation"
        RA1[Collect embedding files]
        RA2[Create unified index]
        RA3[Validate completeness]
        RA4[Vector database format]
        RA1 --> RA2 --> RA3 --> RA4
    end

    subgraph "Performance Metrics"
        PM1[Throughput: ~3K embeddings/min per T4]
        PM2[Total: 100M docs in ~23 days (single T4)]
        PM3[GPU Utilization: 80-90%]
        PM4[Cost: ~$0.0003 per 1K embeddings]
    end

    DS --> DC1
    DC3 --> W1
    DC3 --> W2
    DC3 --> W3
    DC3 --> W4

    W1 --> EG1
    W2 --> EG1
    W3 --> EG1
    W4 --> EG1

    EG4 --> RA1

    style DS fill:#e1f5fe
    style W1 fill:#f3e5f5
    style W2 fill:#f3e5f5
    style W3 fill:#f3e5f5
    style W4 fill:#f3e5f5
    style RA4 fill:#e8f5e8
```

## Scaling Patterns

```mermaid
flowchart TD
    subgraph "Single Node (Baseline)"
        SN[1x T4 GPU<br/>~3K embeddings/min<br/>100M docs = 23 days]
    end

    subgraph "Multi-Node (Recommended)"
        direction LR
        MN1[Node 1<br/>1x T4 GPU<br/>~3K embeddings/min<br/>Chunks 1-50]
        MN2[Node 2<br/>1x T4 GPU<br/>~3K embeddings/min<br/>Chunks 51-100]

        MN1 -.-> RESULT1[Total: ~6K embeddings/min<br/>100M docs = 12 days]
        MN2 -.-> RESULT1
    end

    subgraph "Auto-Scaling (Enterprise)"
        direction TB
        AS[Cluster Auto-Scaler<br/>Scale based on queue depth]

        subgraph "GPU Nodes"
            direction LR
            ASN1[Node 1<br/>1x T4 GPU]
            ASN2[Node 2<br/>1x T4 GPU]
            ASN3[Node 3<br/>1x T4 GPU]
            ASNN[Node N<br/>1x T4 GPU]
        end

        AS --> ASN1
        AS --> ASN2
        AS --> ASN3
        AS --> ASNN

        RESULT2[Target: 100M docs in 24 hours<br/>Required: ~70K embeddings/min<br/>= 24 T4 GPU nodes]
    end

    SN --> MN1
    RESULT1 --> AS

    style SN fill:#ffecb3
    style MN1 fill:#e8f5e8
    style MN2 fill:#e8f5e8
    style RESULT1 fill:#e3f2fd
    style AS fill:#f3e5f5
    style RESULT2 fill:#fff3e0
```

These Mermaid diagrams provide cleaner, more professional visualizations of the GPU workload architecture that will render properly in any markdown-compatible environment.

These diagrams illustrate the key architectural patterns for implementing GPU workloads on Argo Workflows, showing data flow, resource allocation, and scaling strategies for production deployments.
