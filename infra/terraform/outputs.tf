output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_id" {
  value = aws_subnet.public_a.id
}

output "sg_client_id" {
  value = aws_security_group.client.id
}

output "sg_model_id" {
  value = aws_security_group.model.id
}

output "gpu_public_ip" {
  value = data.aws_instance.gpu_live.public_ip
}

output "gpu_private_ip" {
  value = data.aws_instance.gpu_live.private_ip
}

output "cpu_public_ip" {
  value = data.aws_instance.cpu_live.public_ip
}

output "cpu_private_ip" {
  value = data.aws_instance.cpu_live.private_ip
}

output "gpu_instance_id" {
  value = aws_instance.gpu.id
}

output "cpu_instance_id" {
  value = aws_instance.cpu.id
}

output "ssh_config_path" {
  description = "Path to generated SSH config. Add to ~/.ssh/config: Include <this path>"
  value       = abspath("${path.module}/../ssh/aip-intern.conf")
}

output "ssh_usage" {
  description = "Quick SSH commands after adding Include to ~/.ssh/config"
  value       = <<-EOF
    # Add this line to ~/.ssh/config:
    Include ${abspath("${path.module}/../ssh/aip-intern.conf")}

    # Then connect with:
    ssh aip-intern-gpu
    ssh aip-intern-cpu
  EOF
}
