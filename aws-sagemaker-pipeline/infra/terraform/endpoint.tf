resource "aws_sagemaker_model" "this" {
  name               = local.model_name
  execution_role_arn = var.sagemaker_execution_role_arn

  primary_container {
    image          = var.image_uri
    model_data_url = var.model_data_url

    environment = {
      SAGEMAKER_PROGRAM = "mlops_serving_starter/sagemaker/inference.py"
    }
  }
}

# --- Serverless endpoint config ---
resource "aws_sagemaker_endpoint_configuration" "serverless" {
  count = var.use_serverless ? 1 : 0
  name  = "${local.endpoint_config_name}-serverless"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.this.name

    serverless_config {
      max_concurrency   = var.serverless_max_concurrency
      memory_size_in_mb = var.serverless_memory_size_in_mb
    }
  }
}

# --- Real-time endpoint config ---
resource "aws_sagemaker_endpoint_configuration" "realtime" {
  count = var.use_serverless ? 0 : 1
  name  = local.endpoint_config_name

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.this.name
    initial_instance_count = var.endpoint_initial_instance_count
    instance_type          = var.endpoint_instance_type
    initial_variant_weight = 1
  }
}

resource "aws_sagemaker_endpoint" "this" {
  name = local.endpoint_name
  endpoint_config_name = (
    var.use_serverless
    ? aws_sagemaker_endpoint_configuration.serverless[0].name
    : aws_sagemaker_endpoint_configuration.realtime[0].name
  )
}
