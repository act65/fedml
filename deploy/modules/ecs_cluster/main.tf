variable "cluster_name" {}
variable "instance_type" {}
variable "vpc_id" {}
variable "private_subnets" {}
variable "security_group_id" {}
variable "docker_image" {}
variable "service_discovery_namespace" {}
variable "n_workers" {}

resource "aws_ecs_cluster" "cluster" {
  name = var.cluster_name
}

# IAM Role for EC2 Instances
resource "aws_iam_role" "ecs_instance_role" {
  name = "ecsInstanceRole-${var.cluster_name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

# Attach the AmazonEC2ContainerServiceforEC2Role policy
resource "aws_iam_role_policy_attachment" "ecs_instance_role" {
  role       = aws_iam_role.ecs_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ecs" {
  name = "ecsInstanceProfile-${var.cluster_name}"
  role = aws_iam_role.ecs_instance_role.name
}


resource "aws_ecs_capacity_provider" "ec2" {
  name = "ec2-provider"

  auto_scaling_group_provider {
    auto_scaling_group_arn = aws_autoscaling_group.ec2.arn
  }
}

resource "aws_launch_template" "ec2" {
  name_prefix   = "ecs-${var.cluster_name}-"
  image_id      = data.aws_ami.ecs_optimized.id
  instance_type = var.instance_type
  vpc_security_group_ids = [var.security_group_id]

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs.name
  }

  user_data = base64encode(<<-EOF
#!/bin/bash
echo ECS_CLUSTER=${var.cluster_name} >> /etc/ecs/ecs.config
EOF
  )
}


resource "aws_autoscaling_group" "ec2" {
  name                 = "ecs-${var.cluster_name}"
  min_size             = var.n_workers
  max_size             = var.n_workers
  vpc_zone_identifier  = var.private_subnets

  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.ec2.id
        version = "$Latest"
      }
    }
  }

  tag {
    key                 = "Name"
    value               = "ECS-${var.cluster_name}"
    propagate_at_launch = true
  }
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "worker"
  network_mode             = "bridge"
  requires_compatibilities = ["EC2"]
  cpu                      = "256"  # Add CPU allocation
  memory                   = "512"  # Add memory allocation

  container_definitions = jsonencode([{
    name      = "worker",
    image     = var.docker_image,
    essential = true,
    memoryReservation = 512,  # Add memory reservation
    portMappings = [{
      containerPort = 8000,
      hostPort      = 8000
    }],
    environment = [
      { name = "SERVICE_DISCOVERY_ENDPOINT", value = "workers.ml.internal" }
    ]
  }])
}

resource "aws_ecs_service" "worker" {
  name            = "worker-service"
  cluster         = aws_ecs_cluster.cluster.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.n_workers  # Add this if not already present

  # Add network configuration for bridge mode
  network_configuration {
    security_groups = [var.security_group_id]
    subnets         = var.private_subnets
    assign_public_ip = false
  }

  service_registries {
    registry_arn   = var.service_discovery_namespace
    container_name = "worker"
    container_port = 8000
  }

  depends_on = [aws_autoscaling_group.ec2]
}


data "aws_ami" "ecs_optimized" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-ecs-hvm-*-x86_64*"]  # Updated to Amazon Linux 2023
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}