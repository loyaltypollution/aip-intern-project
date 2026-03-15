#!/usr/bin/env bash
# Dynamic inventory — reads IPs from Terraform outputs.
# Usage: ansible-inventory --list  (called automatically by Ansible)

set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../terraform" && pwd)"
SSH_KEY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../ssh" && pwd)/aip-intern-generated-key.pem"

GPU_PUBLIC_IP=$(terraform -chdir="$INFRA_DIR" output -raw gpu_public_ip)
GPU_PRIVATE_IP=$(terraform -chdir="$INFRA_DIR" output -raw gpu_private_ip)
CPU_PUBLIC_IP=$(terraform -chdir="$INFRA_DIR" output -raw cpu_public_ip)

cat <<EOF
{
  "gpu": {
    "hosts": ["${GPU_PUBLIC_IP}"],
    "vars": {}
  },
  "cpu": {
    "hosts": ["${CPU_PUBLIC_IP}"],
    "vars": {}
  },
  "_meta": {
    "hostvars": {
      "${GPU_PUBLIC_IP}": {
        "ansible_user": "ubuntu",
        "ansible_ssh_private_key_file": "${SSH_KEY}",
        "gpu_private_ip": "${GPU_PRIVATE_IP}"
      },
      "${CPU_PUBLIC_IP}": {
        "ansible_user": "ubuntu",
        "ansible_ssh_private_key_file": "${SSH_KEY}",
        "gpu_private_ip": "${GPU_PRIVATE_IP}"
      }
    }
  }
}
EOF
