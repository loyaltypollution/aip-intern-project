variable "cpu_ami_id" {
  description = "CPU AMI ID for ap-northeast-1 (Ubuntu Server 22.04 LTS (HVM), SSD Volume Type)"
  type        = string
  default     = "ami-0d49f1fe982e06148"
}

variable "cpu_instance_type" {
  description = "CPU instance type"
  type        = string
  default     = "c7i.2xlarge"
}

resource "aws_instance" "cpu" {
  ami                         = var.cpu_ami_id
  instance_type               = var.cpu_instance_type
  subnet_id                   = aws_subnet.public_c.id
  vpc_security_group_ids      = [aws_security_group.client.id]
  key_name                    = aws_key_pair.deployer.key_name
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  metadata_options {
    http_tokens = "required"
  }


  lifecycle {
    ignore_changes = [
      associate_public_ip_address,
      private_ip,
      primary_network_interface_id
    ]
  }

  tags = merge(local.tags, { Name = "${local.name_prefix}-cpu" })
}

resource "aws_ec2_instance_state" "cpu_stopped" {
  instance_id = aws_instance.cpu.id
  state       = var.instance_state
}
