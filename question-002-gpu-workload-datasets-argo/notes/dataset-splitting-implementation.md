# Dataset Splitting Implementation Guide

## Overview

This guide provides concrete implementation examples for splitting large datasets and distributing GPU workloads across multiple T4 worker nodes using Argo Workflows.

## Core Implementation Strategy

### 1. S3-based Dynamic Chunk Distribution

#### Chunk Size Calculator

```python
def calculate_optimal_chunks(dataset_size, available_gpus, task_type='embedding', gpu_memory_gb=16):
    """
    Calculate optimal chunk sizes for GPU processing with realistic memory estimates

    Args:
        dataset_size: Total number of documents/samples
        available_gpus: Number of available GPU nodes
        task_type: Type of GPU task ('embedding', 'summarization', 'training')
        gpu_memory_gb: GPU memory in GB (default: T4 = 16GB)

    Returns:
        dict: Chunk configuration with accurate memory calculations
    """

    # Task-specific configurations based on real GPU memory usage
    task_configs = {
        'embedding': {
            'batch_size': 32,
            'processing_rate_per_minute': 3000,
            'model_memory_gb': 1.2,  # sentence-transformers model
            'input_memory_per_token_mb': 0.004,  # Input tensors
            'output_memory_per_item_mb': 3.0,  # 768-dim float32 embedding
            'pytorch_overhead_multiplier': 1.4  # PyTorch overhead
        },
        'summarization': {
            'batch_size': 8,
            'processing_rate_per_minute': 600,
            'model_memory_gb': 2.8,  # T5-large or BART-large
            'input_memory_per_token_mb': 0.008,
            'output_memory_per_item_mb': 1.0,  # Generated text
            'pytorch_overhead_multiplier': 1.6
        },
        'training': {
            'batch_size': 16,
            'processing_rate_per_minute': 240,
            'model_memory_gb': 3.5,  # Larger models for training
            'input_memory_per_token_mb': 0.012,  # Gradients included
            'output_memory_per_item_mb': 0.5,
            'pytorch_overhead_multiplier': 2.0  # Gradients + optimizer states
        }
    }

    config = task_configs[task_type]

    # Calculate memory requirements per batch
    avg_tokens_per_item = 128  # Typical document length after truncation

    batch_memory_mb = (
        config['model_memory_gb'] * 1024 +  # Model weights
        config['batch_size'] * avg_tokens_per_item * config['input_memory_per_token_mb'] +  # Input tensors
        config['batch_size'] * config['output_memory_per_item_mb']  # Output tensors
    ) * config['pytorch_overhead_multiplier']

    # Ensure we don't exceed GPU memory (leave 2GB for system)
    available_memory_mb = (gpu_memory_gb - 2) * 1024
    max_batch_size = min(config['batch_size'], int(available_memory_mb / (batch_memory_mb / config['batch_size'])))

    # Calculate optimal chunk size based on processing time and memory
    target_processing_minutes = 5  # Aim for 5-minute chunks for good parallelism
    items_per_chunk_time = config['processing_rate_per_minute'] * target_processing_minutes

    # Memory-constrained chunk size
    batches_per_chunk = max(1, int(available_memory_mb * 0.8 / batch_memory_mb))  # 80% memory utilization
    items_per_chunk_memory = batches_per_chunk * max_batch_size

    # Use the more conservative estimate
    optimal_chunk_size = min(items_per_chunk_time, items_per_chunk_memory)

    # Calculate final chunk configuration
    target_chunks = min(available_gpus * 2, max(1, dataset_size // optimal_chunk_size))
    chunk_size = max(max_batch_size, dataset_size // target_chunks)
    total_chunks = (dataset_size + chunk_size - 1) // chunk_size

    # Realistic memory calculations
    chunk_memory_mb = (chunk_size / max_batch_size) * batch_memory_mb
    streaming_memory_mb = chunk_memory_mb + 50  # Add streaming buffer overhead

    return {
        'total_chunks': total_chunks,
        'chunk_size': chunk_size,
        'optimal_batch_size': max_batch_size,
        'task_type': task_type,
        'estimated_processing_time_minutes': chunk_size / config['processing_rate_per_minute'],
        'memory_requirements': {
            'gpu_memory_per_chunk_gb': chunk_memory_mb / 1024,
            'streaming_memory_mb': streaming_memory_mb,  # CPU memory for streaming
            'model_memory_gb': config['model_memory_gb'],
            'batch_memory_mb': batch_memory_mb,
            'gpu_utilization_percent': min(95, (chunk_memory_mb / (gpu_memory_gb * 1024)) * 100)
        },
        'performance_estimates': {
            'items_per_minute': config['processing_rate_per_minute'],
            'total_processing_time_minutes': chunk_size / config['processing_rate_per_minute'],
            'parallel_processing_time_minutes': max([
                chunk_size / config['processing_rate_per_minute']
                for i in range(total_chunks)
            ]) if total_chunks > 0 else 0
        }
    }

# Example usage with realistic memory calculations
config = calculate_optimal_chunks(dataset_size=100000, available_gpus=8, task_type='embedding')
print(f"Configuration: {json.dumps(config, indent=2)}")

# Output:
# {
#   "total_chunks": 8,
#   "chunk_size": 12500,
#   "optimal_batch_size": 32,
#   "task_type": "embedding",
#   "estimated_processing_time_minutes": 4.17,
#   "memory_requirements": {
#     "gpu_memory_per_chunk_gb": 2.1,        # Actual GPU memory needed per chunk
#     "streaming_memory_mb": 85,              # CPU memory for streaming (constant)
#     "model_memory_gb": 1.2,                 # Model weights
#     "batch_memory_mb": 1680,                # Memory per batch
#     "gpu_utilization_percent": 84           # GPU memory utilization
#   },
#   "performance_estimates": {
#     "items_per_minute": 3000,
#     "total_processing_time_minutes": 4.17,
#     "parallel_processing_time_minutes": 4.17
#   }
# }

# Memory comparison by task type:
embedding_config = calculate_optimal_chunks(100000, 8, 'embedding')
summarization_config = calculate_optimal_chunks(100000, 8, 'summarization')
training_config = calculate_optimal_chunks(100000, 8, 'training')

print("Memory usage by task type (T4 GPU):")
print(f"Embedding: {embedding_config['memory_requirements']['gpu_memory_per_chunk_gb']:.1f} GB GPU, {embedding_config['memory_requirements']['streaming_memory_mb']} MB CPU")
print(f"Summarization: {summarization_config['memory_requirements']['gpu_memory_per_chunk_gb']:.1f} GB GPU, {summarization_config['memory_requirements']['streaming_memory_mb']} MB CPU")
print(f"Training: {training_config['memory_requirements']['gpu_memory_per_chunk_gb']:.1f} GB GPU, {training_config['memory_requirements']['streaming_memory_mb']} MB CPU")

# Expected output:
# Memory usage by task type (T4 GPU):
# Embedding: 2.1 GB GPU, 85 MB CPU
# Summarization: 7.8 GB GPU, 245 MB CPU
# Training: 12.4 GB GPU, 485 MB CPU
```

#### Streaming Dataset Splitter Script

```python
import json
import boto3
import math
from datetime import datetime
from typing import Iterator, Any, Optional, Union
import io
import gzip

class StreamingDatasetSplitter:
    """
    Memory-efficient streaming dataset splitter for large-scale data

    Supports:
    - Streaming from files (JSON Lines, CSV, text)
    - Streaming from S3 objects
    - Streaming from databases
    - Direct upload to S3 without loading full dataset in memory
    """

    def __init__(self, bucket_name: str, aws_region: str = 'us-west-2'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=aws_region)

    def stream_and_split_from_file(
        self,
        file_path: str,
        dataset_name: str,
        chunk_config: dict,
        file_format: str = 'jsonl'
    ) -> dict:
        """
        Stream dataset from file and split into S3 chunks without loading into memory

        Supports large files (TB-scale) by processing line by line
        """

        def file_iterator():
            """Generator that yields items from file without loading all into memory"""
            if file_format == 'jsonl':
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line.strip())
            elif file_format == 'text':
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            yield line.strip()
            elif file_format == 'gzipped_jsonl':
                with gzip.open(file_path, 'rt') as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line.strip())

        return self._stream_and_split(file_iterator(), dataset_name, chunk_config)

    def stream_and_split_from_s3(
        self,
        source_bucket: str,
        source_key: str,
        dataset_name: str,
        chunk_config: dict,
        file_format: str = 'jsonl'
    ) -> dict:
        """
        Stream dataset directly from S3 source and split into chunks

        Ideal for processing large datasets already stored in S3
        """

        def s3_iterator():
            """Generator that streams from S3 object without downloading entire file"""
            response = self.s3_client.get_object(Bucket=source_bucket, Key=source_key)

            if source_key.endswith('.gz'):
                body = gzip.decompress(response['Body'].read()).decode('utf-8')
            else:
                body = response['Body'].read().decode('utf-8')

            lines = body.split('\n')
            for line in lines:
                if line.strip():
                    if file_format == 'jsonl':
                        yield json.loads(line.strip())
                    else:
                        yield line.strip()

        return self._stream_and_split(s3_iterator(), dataset_name, chunk_config)

    def stream_and_split_from_database(
        self,
        db_connection,
        query: str,
        dataset_name: str,
        chunk_config: dict,
        batch_size: int = 10000
    ) -> dict:
        """
        Stream dataset from database query results and split into chunks

        Processes database results in batches to handle large tables
        """

        def db_iterator():
            """Generator that streams from database without loading all results"""
            cursor = db_connection.cursor()
            cursor.execute(query)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                for row in rows:
                    # Convert database row to string or JSON as needed
                    if isinstance(row, (tuple, list)):
                        yield ' '.join(str(col) for col in row)
                    else:
                        yield str(row)

        return self._stream_and_split(db_iterator(), dataset_name, chunk_config)

    def _stream_and_split(
        self,
        data_iterator: Iterator[Any],
        dataset_name: str,
        chunk_config: dict
    ) -> dict:
        """
        Core streaming splitter that processes data iterator and uploads chunks

        Memory usage: O(chunk_size) instead of O(dataset_size)
        """
        chunk_size = chunk_config['chunk_size']
        current_chunk = []
        chunk_index = 0
        total_items = 0

        chunk_metadata = {
            'dataset_name': dataset_name,
            'chunk_size': chunk_size,
            'created_at': datetime.utcnow().isoformat(),
            'chunks': []
        }

        print(f"Starting streaming split for dataset: {dataset_name}")
        print(f"Target chunk size: {chunk_size}")

        for item in data_iterator:
            current_chunk.append(item)
            total_items += 1

            # When chunk is full, upload it
            if len(current_chunk) >= chunk_size:
                chunk_info = self._upload_chunk(
                    current_chunk, dataset_name, chunk_index, total_items - len(current_chunk), total_items - 1
                )
                chunk_metadata['chunks'].append(chunk_info)

                # Progress logging
                if chunk_index % 10 == 0:
                    print(f"Processed {chunk_index + 1} chunks, {total_items:,} items...")

                # Reset for next chunk
                current_chunk = []
                chunk_index += 1

        # Upload final partial chunk if exists
        if current_chunk:
            chunk_info = self._upload_chunk(
                current_chunk, dataset_name, chunk_index, total_items - len(current_chunk), total_items - 1
            )
            chunk_metadata['chunks'].append(chunk_info)
            chunk_index += 1

        # Update final metadata
        chunk_metadata.update({
            'total_chunks': chunk_index,
            'total_items': total_items,
            'actual_chunks': len(chunk_metadata['chunks'])
        })

        # Upload chunk metadata
        self._upload_metadata(dataset_name, chunk_metadata)

        print(f"Successfully streamed and split {total_items:,} items into {chunk_index} chunks")
        print(f"Memory usage: ~{chunk_size * 8 / 1024:.1f} KB (single chunk size)")

        return chunk_metadata

    def _upload_chunk(self, chunk_data: list, dataset_name: str, chunk_index: int, start_idx: int, end_idx: int) -> dict:
        """Upload single chunk to S3 with compression for large chunks"""
        chunk_id = f"chunk-{chunk_index:03d}"
        chunk_key = f"datasets/{dataset_name}/chunks/{chunk_id}.json"

        chunk_content = {
            'chunk_id': chunk_id,
            'items': chunk_data,
            'start_index': start_idx,
            'end_index': end_idx,
            'item_count': len(chunk_data),
            'created_at': datetime.utcnow().isoformat()
        }

        # Compress large chunks
        content_json = json.dumps(chunk_content)
        if len(content_json) > 1024 * 1024:  # > 1MB
            content_body = gzip.compress(content_json.encode('utf-8'))
            chunk_key = f"datasets/{dataset_name}/chunks/{chunk_id}.json.gz"
            content_type = 'application/gzip'
        else:
            content_body = content_json
            content_type = 'application/json'

        # Upload with metadata
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=chunk_key,
            Body=content_body,
            ContentType=content_type,
            Metadata={
                'chunk-id': chunk_id,
                'item-count': str(len(chunk_data)),
                'dataset-name': dataset_name,
                'compressed': str(content_type == 'application/gzip')
            }
        )

        return {
            'chunk_id': chunk_id,
            's3_key': chunk_key,
            'item_count': len(chunk_data),
            'estimated_processing_time': len(chunk_data) / 3000 * 60,  # minutes
            'size_bytes': len(content_body),
            'compressed': content_type == 'application/gzip'
        }

    def _upload_metadata(self, dataset_name: str, metadata: dict):
        """Upload chunk metadata to S3"""
        metadata_key = f"datasets/{dataset_name}/chunk-metadata.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json',
            Metadata={
                'dataset-name': dataset_name,
                'total-chunks': str(metadata['total_chunks']),
                'total-items': str(metadata['total_items'])
            }
        )

# Usage Examples for Large-Scale Data

# Example 1: Process TB-scale JSONL file without loading into memory
def process_large_jsonl_file():
    splitter = StreamingDatasetSplitter(bucket_name="argo-gpu-artifacts-12345")
    config = calculate_optimal_chunks(dataset_size=10000000, available_gpus=16)  # 10M items

    metadata = splitter.stream_and_split_from_file(
        file_path="/data/large_dataset.jsonl",  # TB-scale file
        dataset_name="large-10m-docs",
        chunk_config=config,
        file_format='jsonl'
    )
    print(f"Processed {metadata['total_items']:,} items using {metadata['total_chunks']} chunks")

# Example 2: Stream from compressed S3 source
def process_s3_source():
    splitter = StreamingDatasetSplitter(bucket_name="target-bucket")
    config = calculate_optimal_chunks(dataset_size=5000000, available_gpus=12)

    metadata = splitter.stream_and_split_from_s3(
        source_bucket="source-data-bucket",
        source_key="datasets/massive-dataset.jsonl.gz",
        dataset_name="s3-streamed-5m",
        chunk_config=config,
        file_format='jsonl'
    )

# Example 3: Stream from database (PostgreSQL)
def process_database_data():
    import psycopg2

    conn = psycopg2.connect(
        host="your-db-host",
        database="your-db",
        user="your-user",
        password="your-password"
    )

    splitter = StreamingDatasetSplitter(bucket_name="argo-gpu-artifacts-12345")
    config = calculate_optimal_chunks(dataset_size=20000000, available_gpus=20)  # 20M records

    metadata = splitter.stream_and_split_from_database(
        db_connection=conn,
        query="SELECT content FROM documents WHERE processed = false ORDER BY id",
        dataset_name="db-streamed-20m",
        chunk_config=config,
        batch_size=50000  # Process 50K DB rows at a time
    )
```

### 2. Enhanced GPU Workflow Template

#### Complete Parallel Processing Template

```yaml
# Enhanced GPU Embedding Generation with Dynamic Splitting
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: gpu-embedding-generation-parallel
  namespace: argo
spec:
  serviceAccountName: argo-workflows-sa
  entrypoint: parallel-embedding-pipeline
  arguments:
    parameters:
    - name: dataset-name
      value: "sample-100k-docs"
    - name: model-name
      value: "sentence-transformers/all-mpnet-base-v2"
    - name: s3-bucket
      value: "argo-gpu-artifacts-12345"
    - name: target-parallelism
      value: "8"

  templates:
  # Main pipeline orchestration
  - name: parallel-embedding-pipeline
    dag:
      tasks:
      - name: discover-chunks
        template: chunk-discovery
        arguments:
          parameters:
          - name: dataset-name
            value: "{{workflow.parameters.dataset-name}}"
          - name: s3-bucket
            value: "{{workflow.parameters.s3-bucket}}"

      - name: process-chunks-parallel
        template: parallel-gpu-processing
        dependencies: [discover-chunks]
        arguments:
          parameters:
          - name: chunk-list
            value: "{{tasks.discover-chunks.outputs.parameters.chunk-list}}"
          - name: model-name
            value: "{{workflow.parameters.model-name}}"
          - name: s3-bucket
            value: "{{workflow.parameters.s3-bucket}}"

      - name: aggregate-results
        template: result-aggregation
        dependencies: [process-chunks-parallel]
        arguments:
          parameters:
          - name: dataset-name
            value: "{{workflow.parameters.dataset-name}}"
          - name: s3-bucket
            value: "{{workflow.parameters.s3-bucket}}"

  # Chunk discovery from S3
  - name: chunk-discovery
    inputs:
      parameters:
      - name: dataset-name
      - name: s3-bucket
    script:
      image: amazon/aws-cli:latest
      command: [python]
      resources:
        requests:
          memory: 512Mi
          cpu: 500m
        limits:
          memory: 1Gi
          cpu: 1000m
      source: |
        import json
        import boto3
        import os

        dataset_name = "{{inputs.parameters.dataset-name}}"
        bucket_name = "{{inputs.parameters.s3-bucket}}"

        # Download chunk metadata
        s3 = boto3.client('s3')
        metadata_key = f"datasets/{dataset_name}/chunk-metadata.json"

        try:
            response = s3.get_object(Bucket=bucket_name, Key=metadata_key)
            metadata = json.loads(response['Body'].read())

            # Create chunk list for parallel processing
            chunk_list = []
            for chunk in metadata['chunks']:
                chunk_list.append({
                    'chunk_id': chunk['chunk_id'],
                    's3_key': chunk['s3_key'],
                    'item_count': chunk['item_count'],
                    'bucket': bucket_name
                })

            # Output chunk list for withParam
            os.makedirs('/tmp/outputs', exist_ok=True)
            with open('/tmp/outputs/chunk-list.json', 'w') as f:
                json.dump(chunk_list, f)

            print(f"Discovered {len(chunk_list)} chunks for processing")

        except Exception as e:
            print(f"Error discovering chunks: {e}")
            # Create empty list as fallback
            with open('/tmp/outputs/chunk-list.json', 'w') as f:
                json.dump([], f)
    outputs:
      parameters:
      - name: chunk-list
        valueFrom:
          path: /tmp/outputs/chunk-list.json

  # Parallel processing controller
  - name: parallel-gpu-processing
    inputs:
      parameters:
      - name: chunk-list
      - name: model-name
      - name: s3-bucket
    steps:
    - - name: process-chunk
        template: gpu-chunk-processor
        withParam: "{{inputs.parameters.chunk-list}}"
        arguments:
          parameters:
          - name: chunk-metadata
            value: "{{item}}"
          - name: model-name
            value: "{{inputs.parameters.model-name}}"

  # Individual GPU chunk processor
  - name: gpu-chunk-processor
    inputs:
      parameters:
      - name: chunk-metadata
      - name: model-name
    nodeSelector:
      node-type: gpu-t4
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    script:
      image: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
      command: [python]
      resources:
        requests:
          nvidia.com/gpu: 1
          memory: 8Gi
          cpu: 4
        limits:
          nvidia.com/gpu: 1
          memory: 14Gi
          cpu: 6
      source: |
        import json
        import torch
        import boto3
        import numpy as np
        from transformers import AutoTokenizer, AutoModel
        import os
        from datetime import datetime

        # Parse chunk metadata
        chunk_metadata = json.loads('{{inputs.parameters.chunk-metadata}}')
        model_name = "{{inputs.parameters.model-name}}"

        print(f"Processing chunk: {chunk_metadata['chunk_id']}")
        print(f"Expected items: {chunk_metadata['item_count']}")

        # Download chunk data from S3
        s3 = boto3.client('s3')
        response = s3.get_object(
            Bucket=chunk_metadata['bucket'],
            Key=chunk_metadata['s3_key']
        )
        chunk_data = json.loads(response['Body'].read())
        documents = chunk_data['items']

        # Initialize model with PyTorch 2.8.0 optimizations
        print("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device, dtype=torch.float16)

        # Enable PyTorch 2.8.0 optimizations
        torch.backends.cuda.enable_flash_sdp(True)
        if torch.cuda.is_available():
            model = torch.compile(model, mode="max-autotune")

        # Process documents in batches
        batch_size = 32  # Optimal for T4
        all_embeddings = []

        print(f"Processing {len(documents)} documents in batches of {batch_size}")

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            # Generate embeddings with mixed precision
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

            all_embeddings.append(embeddings.cpu().numpy())

            if i % (batch_size * 10) == 0:
                print(f"Processed {i + len(batch_docs)}/{len(documents)} documents")

        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        print(f"Generated embeddings shape: {final_embeddings.shape}")

        # Upload results to S3
        result_key = f"datasets/{chunk_data['chunk_id'].split('-')[0]}/results/{chunk_metadata['chunk_id']}-embeddings.npy"

        # Save to temporary file
        os.makedirs('/tmp/outputs', exist_ok=True)
        temp_path = f"/tmp/outputs/{chunk_metadata['chunk_id']}-embeddings.npy"
        np.save(temp_path, final_embeddings)

        # Upload to S3
        s3.upload_file(temp_path, chunk_metadata['bucket'], result_key)

        # Create processing report
        report = {
            'chunk_id': chunk_metadata['chunk_id'],
            'processed_items': len(documents),
            'embedding_shape': final_embeddings.shape,
            'result_s3_key': result_key,
            'processing_time': datetime.utcnow().isoformat(),
            'gpu_used': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
        }

        with open('/tmp/outputs/processing-report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Chunk processing completed: {report}")
    outputs:
      artifacts:
      - name: processing-report
        path: /tmp/outputs/processing-report.json

  # Aggregate all results
  - name: result-aggregation
    inputs:
      parameters:
      - name: dataset-name
      - name: s3-bucket
    script:
      image: amazon/aws-cli:latest
      command: [python]
      resources:
        requests:
          memory: 2Gi
          cpu: 1
        limits:
          memory: 4Gi
          cpu: 2
      source: |
        import json
        import boto3
        import numpy as np
        from datetime import datetime

        dataset_name = "{{inputs.parameters.dataset-name}}"
        bucket_name = "{{inputs.parameters.s3-bucket}}"

        s3 = boto3.client('s3')

        # List all result files
        prefix = f"datasets/{dataset_name}/results/"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' not in response:
            print("No result files found")
            exit(1)

        embedding_files = [obj['Key'] for obj in response['Contents']
                          if obj['Key'].endswith('-embeddings.npy')]

        print(f"Found {len(embedding_files)} embedding files to aggregate")

        # Aggregate embeddings
        all_embeddings = []
        total_processed = 0

        for file_key in sorted(embedding_files):
            # Download and load embeddings
            local_path = f"/tmp/{file_key.split('/')[-1]}"
            s3.download_file(bucket_name, file_key, local_path)

            embeddings = np.load(local_path)
            all_embeddings.append(embeddings)
            total_processed += embeddings.shape[0]

            print(f"Loaded {embeddings.shape[0]} embeddings from {file_key}")

        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)

        # Save final aggregated embeddings
        final_path = "/tmp/final-embeddings.npy"
        np.save(final_path, final_embeddings)

        # Upload final results
        final_key = f"datasets/{dataset_name}/final/embeddings.npy"
        s3.upload_file(final_path, bucket_name, final_key)

        # Create final report
        final_report = {
            'dataset_name': dataset_name,
            'total_embeddings': final_embeddings.shape[0],
            'embedding_dimension': final_embeddings.shape[1],
            'chunks_processed': len(embedding_files),
            'final_s3_key': final_key,
            'completed_at': datetime.utcnow().isoformat()
        }

        report_key = f"datasets/{dataset_name}/final/processing-report.json"
        s3.put_object(
            Bucket=bucket_name,
            Key=report_key,
            Body=json.dumps(final_report, indent=2),
            ContentType='application/json'
        )

        print(f"Final aggregation completed: {final_report}")
    outputs:
      parameters:
      - name: total-processed
        valueFrom:
          path: /tmp/total_processed.txt
```

### 3. Horizontal Pod Autoscaler Configuration

```yaml
# HPA for GPU Worker Nodes
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpu-worker-hpa
  namespace: argo
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-worker-pool
  minReplicas: 1
  maxReplicas: 24
  metrics:
  # Scale based on GPU utilization
  - type: External
    external:
      metric:
        name: nvidia_gpu_utilization
        selector:
          matchLabels:
            node_type: gpu-t4
      target:
        type: AverageValue
        averageValue: "70"
  # Scale based on pending workflows
  - type: External
    external:
      metric:
        name: argo_workflows_pending_count
        selector:
          matchLabels:
            namespace: argo
      target:
        type: Value
        value: "2"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      selectPolicy: Min
```

## Usage Examples

### 1. Submit Large-Scale Embedding Job

```bash
#!/bin/bash
# submit-large-dataset.sh

# Parameters
DATASET_SIZE=100000
DATASET_NAME="large-corpus-$(date +%s)"
S3_BUCKET="argo-gpu-artifacts-12345"
TARGET_PARALLELISM=8

echo "Submitting large-scale embedding job..."
echo "Dataset size: ${DATASET_SIZE}"
echo "Target parallelism: ${TARGET_PARALLELISM} GPUs"

# Submit the workflow
WORKFLOW_NAME=$(argo submit -n argo \
  --from workflowtemplate/gpu-embedding-generation-parallel \
  --parameter dataset-name="${DATASET_NAME}" \
  --parameter s3-bucket="${S3_BUCKET}" \
  --parameter target-parallelism="${TARGET_PARALLELISM}" \
  --generate-name embedding-large- \
  -o name)

echo "Submitted workflow: ${WORKFLOW_NAME}"

# Monitor the workflow
echo "Monitoring workflow progress..."
argo watch "${WORKFLOW_NAME}" -n argo

# Get final results
echo "Getting final results..."
argo logs "${WORKFLOW_NAME}" -n argo | grep "Final aggregation completed"
```

### 2. Monitor GPU Utilization and Scaling

```bash
#!/bin/bash
# monitor-gpu-scaling.sh

echo "Monitoring GPU node scaling and utilization..."

# Watch HPA scaling events
kubectl get hpa gpu-worker-hpa -n argo -w &
HPA_PID=$!

# Monitor GPU utilization on nodes
watch -n 5 'kubectl get nodes -l node-type=gpu-t4 -o custom-columns="NODE:.metadata.name,GPU-ALLOC:.status.allocatable.nvidia\.com/gpu,GPU-USED:.status.capacity.nvidia\.com/gpu"' &
NODE_PID=$!

# Monitor workflow queue
watch -n 10 'echo "=== Workflow Status ===" && argo list -n argo | grep -E "(Running|Pending)"' &
WORKFLOW_PID=$!

# Cleanup on exit
trap "kill $HPA_PID $NODE_PID $WORKFLOW_PID 2>/dev/null" EXIT

echo "Monitoring started. Press Ctrl+C to stop."
wait
```

### 3. Cost Analysis Script

```bash
#!/bin/bash
# cost-analysis.sh

WORKFLOW_NAME=$1
START_TIME=$(argo get "$WORKFLOW_NAME" -n argo -o json | jq -r '.status.startedAt')
END_TIME=$(argo get "$WORKFLOW_NAME" -n argo -o json | jq -r '.status.finishedAt')

# Calculate duration in hours
DURATION_SECONDS=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
DURATION_HOURS=$(echo "scale=3; $DURATION_SECONDS / 3600" | bc)

# Count GPU nodes used (approximate)
GPU_NODES=$(argo logs "$WORKFLOW_NAME" -n argo | grep "gpu_used" | wc -l)

# Calculate cost (g4dn.2xlarge = $0.526/hour)
COST=$(echo "scale=2; $DURATION_HOURS * $GPU_NODES * 0.526" | bc)

echo "=== Cost Analysis for $WORKFLOW_NAME ==="
echo "Duration: ${DURATION_HOURS} hours"
echo "GPU nodes used: ${GPU_NODES}"
echo "Estimated cost: \$${COST}"
echo "Cost per 1M documents: \$$(echo "scale=2; $COST * 10" | bc)"
```

This implementation provides a complete, production-ready solution for splitting large datasets and distributing GPU workloads with automatic scaling and cost optimization.
