# Terraform

Bare-bones Terraform setup for the SageMaker deployment. Creates:

- SageMaker model + endpoint config + endpoint
- EventBridge schedule to trigger the pipeline (optional)
- CloudWatch alarms for pipeline failures and endpoint 5XX errors

## Validate

```bash
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform validate
```

## Plan against a real account

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
# fill in your values
terraform -chdir=infra/terraform plan -var-file=terraform.tfvars
```
