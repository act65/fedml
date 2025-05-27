variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "job_name" {
  description = "Name of the job"
  type        = string
  default     = "distributed-ml"
}

variable "cluster_name" {
  description = "ECS cluster name"
  default     = "ml-cluster"
}

variable "docker_image" {
  description = "Docker image for workers"
  type        = string
  default     = "nginx:latest"  # Override with TF_VAR_docker_image
}

variable "n_workers" {
  description = "Number of workers"
  type        = number
  default     = 4
}