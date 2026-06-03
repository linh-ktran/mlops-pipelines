# MLOps Pipelines

A collection of ML pipeline experiments to learn and test different MLOps platforms and techniques.

## Projects

| Project | Platform / Technique | Description |
|---------|---------------------|-------------|
| `aws-sagemaker-pipeline/` | AWS SageMaker + MLflow | End-to-end ML pipeline with SageMaker endpoints, Terraform infra |
| `ibm-code-engine-pipeline/` | IBM Code Engine + COS | ML pipeline using IBM Cloud Code Engine and Cloud Object Storage |

## Naming convention

Projects are named by **platform/technique**, not by use case:
`<platform>-<technique>-pipeline`

Examples: `ibm-watsonx-ml-pipeline`, `aws-sagemaker-pipeline`, `vertex-ai-pipeline`

## Adding a new project

Simply create a new directory at the root with its own `pyproject.toml` and dependencies.
Each sub-project is self-contained with its own virtual environment.

