output "ecr_repository_url" {
  value = aws_ecr_repository.image_repo.repository_url
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.logs.name
}