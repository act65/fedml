provider "aws" {
  region = var.region
}

# ECR Repository
resource "aws_ecr_repository" "image_repo" {
  name = var.docker_image
  force_delete = true
}

data "aws_caller_identity" "current" {}

resource "aws_iam_policy" "terraform_execution" {
  name        = "TerraformExecutionPolicy"
  description = "Minimum permissions for Terraform"
  
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:CreateRepository", "ecr:DeleteRepository", "ecr:DescribeRepositories",
          "ecs:CreateCluster", "ecs:DescribeClusters", "ecs:DeleteCluster",
          "logs:CreateLogGroup", "logs:DescribeLogGroups", "logs:DeleteLogGroup",
          "iam:CreateRole", "iam:AttachRolePolicy",
          "ec2:DescribeVpcs", "ec2:CreateSecurityGroup",  # Narrow down permissions
          "vpc:CreateTags"  # Be specific
        ],
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "terraform_execution" {
  name = "TerraformExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/weitao"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "terraform" {
  role       = aws_iam_role.terraform_execution.name
  policy_arn = aws_iam_policy.terraform_execution.arn
}


module "network" {
  source          = "./modules/network"
  vpc_cidr        = "10.0.0.0/16"
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.3.0/24", "10.0.4.0/24"]
  cluster_name    = var.cluster_name
  region          = var.region
}

module "service_discovery" {
  source       = "./modules/service_discovery"
  vpc_id       = module.network.vpc_id
  namespace    = "ml.internal"
  service_name = "workers"
}

module "ecs_cluster" {
  source             = "./modules/ecs_cluster"
  cluster_name       = var.cluster_name
  instance_type      = "t3.nano"
  n_workers          = var.n_workers
  vpc_id             = module.network.vpc_id
  private_subnets    = module.network.private_subnets
  security_group_id  = module.network.worker_sg_id
  docker_image       = var.docker_image
  service_discovery_namespace = module.service_discovery.namespace_arn
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "logs" {
  name = "/ecs/${var.job_name}"
}

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.65"  # Required for VPC module 5.x
    }
  }
}