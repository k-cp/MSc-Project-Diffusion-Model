#!/bin/bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=00:02:00
#SBATCH --output=testnet_%j.log
curl -s -m 15 -d "compute-node test $(hostname)" https://ntfy.sh/kaya-si-7h3k9x \
  && echo "COMPUTE NODE HAS INTERNET" || echo "COMPUTE NODE BLOCKED"
