#!/usr/bin/env bash
# Downloads the LoCoMo benchmark (snap-research/locomo) into data/locomo/.
set -euo pipefail
mkdir -p "$(dirname "$0")/../data/locomo"
curl -sSL -o "$(dirname "$0")/../data/locomo/locomo10.json" \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
echo "saved data/locomo/locomo10.json"
