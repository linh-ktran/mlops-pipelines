output "sagemaker_endpoint_name" {
  value       = aws_sagemaker_endpoint.this.name
  description = "Created SageMaker endpoint name"
}

output "sagemaker_model_name" {
  value       = aws_sagemaker_model.this.name
  description = "Created SageMaker model name"
}

output "sagemaker_endpoint_config_name" {
  value = (
    var.use_serverless
    ? aws_sagemaker_endpoint_configuration.serverless[0].name
    : aws_sagemaker_endpoint_configuration.realtime[0].name
  )
  description = "Created SageMaker endpoint config name"
}

output "endpoint_type" {
  value       = var.use_serverless ? "serverless" : "realtime"
  description = "Whether this is a serverless or real-time endpoint"
}

output "scheduler_name" {
  value       = var.schedule_enabled ? aws_scheduler_schedule.pipeline_schedule[0].name : null
  description = "Scheduler name when scheduling is enabled"
}
