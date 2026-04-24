variable "gpu_ami_id" {
  description = "GPU AMI ID for ap-northeast-1 (Deep Learning Base AMI with Single CUDA (Ubuntu 22.04))"
  type        = string
  default = "ami-0060f9bdc9af6729d"
}

variable "gpu_instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "g6e.12xlarge"
}

resource "aws_instance" "gpu" {
  ami                         = var.gpu_ami_id
  instance_type               = var.gpu_instance_type
  subnet_id                   = aws_subnet.public_a.id
  vpc_security_group_ids      = [aws_security_group.model.id]
  key_name                    = aws_key_pair.deployer.key_name
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 300
    volume_type = "gp3"
  }

  metadata_options {
    http_tokens = "required"
  }


  lifecycle {
    ignore_changes = [
      associate_public_ip_address,
      private_ip,
      public_dns,
      primary_network_interface_id
    ]
  }

  tags = merge(local.tags, { Name = "${local.name_prefix}-gpu" })
}

resource "aws_ec2_instance_state" "gpu_stopped" {
  instance_id = aws_instance.gpu.id
  state       = var.instance_state
}
