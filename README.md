# News Classification with DistilBERT on AWS SageMaker

A complete end-to-end machine learning pipeline for classifying news headlines into four categories (Business, Technology, Entertainment, Health) using a fine-tuned DistilBERT model deployed on AWS SageMaker with API Gateway integration.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Load Testing](#load-testing)
- [API Gateway Integration](#api-gateway-integration)
- [Usage](#usage)

## Project Overview

This project demonstrates how to:

1. Perform exploratory data analysis on the News Aggregator Dataset
2. Fine-tune a DistilBERT model for multi-class text classification
3. Train the model using AWS SageMaker managed training jobs
4. Deploy the model to a SageMaker real-time inference endpoint
5. Conduct load testing to determine optimal instance sizing
6. Create an AWS Lambda function to invoke the endpoint
7. Expose the model via AWS API Gateway as a REST API

## Dataset

**News Aggregator Dataset** from UCI Machine Learning Repository

The dataset contains news headlines collected from various sources, categorized into four classes:

| Category | Label | Code |
|----------|-------|------|
| Business | b | 0 |
| Technology | t | 1 |
| Entertainment | e | 2 |
| Health | m | 3 |

**Dataset Statistics:**
- Total samples (after deduplication): ~400,000 headlines
- Training subset used: 50% of data (~200,000 samples)
- Train/Test split: 80/20

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│ API Gateway │───▶│   Lambda    │───▶│  SageMaker  │
│  (Request)  │    │   (REST)    │    │  Function   │    │  Endpoint   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                         ┌─────────────┐
                                                         │ DistilBERT  │
                                                         │   Model     │
                                                         └─────────────┘
```

## Project Structure

```
├── README.md
├── TrainingNotebook.ipynb      # SageMaker training job notebook
├── InferenceNotebook.ipynb     # Model deployment and testing notebook
├── scripts/
│   ├── script.py               # Training script for SageMaker
│   ├── inference.py            # Custom inference handlers
│   └── load_test.py            # Load testing utilities
│   └── lambda_function.py      # AWS Lambda handler
└── data/
    └── inputs.tar.gz           # Test input data
```

## Setup & Installation

### Prerequisites

- AWS Account with SageMaker access
- Python 3.11+
- AWS CLI configured with appropriate permissions

### Dependencies

```bash
pip install transformers torch sagemaker pandas numpy tqdm boto3
```

### AWS Permissions Required

- `sagemaker:CreateTrainingJob`
- `sagemaker:CreateModel`
- `sagemaker:CreateEndpointConfig`
- `sagemaker:CreateEndpoint`
- `sagemaker:InvokeEndpoint`
- `lambda:CreateFunction`
- `apigateway:*`
- `s3:GetObject`, `s3:PutObject`

## Exploratory Data Analysis

The dataset preprocessing includes:

```python
# Load and preprocess data
df = pd.read_csv(s3_path, sep="\t", 
                 names=["TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"])
df = df[["TITLE", "CATEGORY"]]
df = df.drop_duplicates()

# Map categories to readable labels
df["CATEGORY"] = df["CATEGORY"].map({
    "b": "Business",
    "t": "Technology",
    "e": "Entertainment",
    "m": "Health"
})

# Encode categories for training
df["ENCODE_CAT"] = df["CATEGORY"].map({
    "Business": 0,
    "Technology": 1,
    "Entertainment": 2,
    "Health": 3
})
```

## Model Training

### Model Architecture

The model extends DistilBERT with a custom classification head:

```python
class DistilBERTModel(torch.nn.Module):
    def __init__(self, model_chkpt):
        super().__init__()
        self.layer1 = AutoModel.from_pretrained(model_chkpt)
        self.feed_forward = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 4)
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Train Batch Size | 4 |
| Valid Batch Size | 2 |
| Learning Rate | 1e-4 |
| Max Sequence Length | 512 |
| Instance Type | ml.g5.2xlarge |

### Launch Training Job

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point="script.py",
    source_dir="./scripts",
    py_version="py311",
    transformers_version="4.49",
    pytorch_version="2.5",
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    output_path="s3://your-bucket/output/",
    hyperparameters={
        'epochs': 3,
        'train_batch_size': 4,
        'valid_batch_size': 2,
        'learning_rate': 1e-04,
        'max_len': 512
    }
)

huggingface_estimator.fit(job_name="finetuning-distilbert-news")
```

## Model Deployment

### Deploy to SageMaker Endpoint

```python
from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    role=role,
    model_data="s3://your-bucket/output/model.tar.gz",
    py_version="py312",
    transformers_version="4.49",
    pytorch_version="2.6",
    entry_point="inference.py",
    source_dir="./scripts",
    name="DistilBert-news-classification-model"
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="Multiclass-NewsTitle-classification-endpoint"
)
```

### Test the Endpoint

```python
data = {"inputs": "The new OnePlus 15 is the phone to beat in 2025"}
prediction = predictor.predict(data)
# Output: {"predicted_label": "Technology", "probabilities": [[0.007, 0.981, 0.009, 0.004]]}
```

## Load Testing

The `load_test.py` script generates test payloads for benchmarking:

```python
input_data = [
    {"inputs": "The new OnePlus 15 is the phone to beat in 2025"},
    {"inputs": "Tesla's latest earnings report signals a major shift in the EV market"},
    {"inputs": "Breakthrough weight-loss drug shows promising results in clinical trials"},
    {"inputs": "Marvel's next phase could redefine the superhero genre"},
    {"inputs": "Taylor Swift's record-breaking tour cements her as the decade's biggest artist"}
]
```


## API Gateway Integration

### Lambda Function

The Lambda function acts as a bridge between API Gateway and SageMaker:

```python
import json
import boto3

def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    body = json.loads(event['body'])
    headline = body['query']['headline']
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='Multiclass-NewsTitle-classification-final-endpoint01',
        ContentType='application/json',
        Body=json.dumps({'inputs': headline})
    )
    
    results = json.loads(response['Body'].read().decode())
    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }
```

### API Gateway Setup

1. Create a new REST API in API Gateway
2. Create a POST method on a `/classify` resource
3. Set Lambda function as the integration target
4. Enable CORS if needed
5. Deploy to a stage (e.g., `prod`)

### API Request Format

```bash
curl -X POST https://your-api-id.execute-api.region.amazonaws.com/prod/classify \
  -H "Content-Type: application/json" \
  -d '{"query": {"headline": "Apple announces new iPhone 16 with AI features"}}'
```

### API Response Format

```json
{
  "predicted_label": "Technology",
  "probabilities": [[0.012, 0.943, 0.028, 0.017]]
}
```

## Usage

### Direct Endpoint Invocation

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName='Multiclass-NewsTitle-classification-final-endpoint01',
    ContentType='application/json',
    Body=json.dumps({'inputs': 'Your headline here'})
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Via API Gateway

```python
import requests

url = "https://your-api-id.execute-api.region.amazonaws.com/prod/classify"
payload = {"query": {"headline": "Your headline here"}}

response = requests.post(url, json=payload)
print(response.json())
```



## Acknowledgments

- Hugging Face Transformers library
- AWS SageMaker documentation
- UCI Machine Learning Repository for the dataset
