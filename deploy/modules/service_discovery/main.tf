variable "vpc_id" {}
variable "namespace" {}
variable "service_name" {}

resource "aws_service_discovery_private_dns_namespace" "ml" {
  name        = var.namespace
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "workers" {
  name = var.service_name

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.ml.id

    dns_records {
      ttl  = 10
      type = "SRV"  # Changed from A to SRV
    }

    routing_policy = "MULTIVALUE"
  }

  # Required for SRV records with bridge networking
  health_check_custom_config {
    failure_threshold = 1
  }
}


output "namespace_arn" {
  value = aws_service_discovery_service.workers.arn
}