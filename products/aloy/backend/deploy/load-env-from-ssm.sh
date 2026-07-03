#!/usr/bin/env bash
# Fetch all SecureString parameters under an SSM path and write them to a
# KEY='VALUE' env file consumable by docker-compose `env_file:`.
#
#   load-env-from-ssm.sh [SSM_PATH] [OUT_FILE]
#
# Defaults: /pori-cloud/prod → /opt/pori-cloud/.env
# Requires the AWS CLI and an IAM role (or credentials) with
# ssm:GetParametersByPath + kms:Decrypt on the SSM KMS key.

set -euo pipefail

PREFIX="${1:-/pori-cloud/prod}"
OUT="${2:-/opt/pori-cloud/.env}"
REGION="${AWS_REGION:-eu-west-1}"

umask 077
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

aws ssm get-parameters-by-path \
    --region "$REGION" \
    --path "$PREFIX" \
    --with-decryption \
    --recursive \
    --query 'Parameters[*].[Name,Value]' \
    --output text \
  | while IFS=$'\t' read -r name value; do
      [ -z "$name" ] && continue
      key="${name##*/}"
      # Single-quote the value and escape any embedded single quotes.
      escaped=$(printf %s "$value" | sed "s/'/'\\\\''/g")
      printf "%s='%s'\n" "$key" "$escaped"
    done > "$tmp"

if [ ! -s "$tmp" ]; then
  echo "No parameters found under $PREFIX in $REGION" >&2
  exit 1
fi

install -m 0600 "$tmp" "$OUT"
echo "Wrote $(wc -l < "$OUT") vars from $PREFIX to $OUT"
