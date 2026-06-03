#!/usr/bin/env bash
# Upload sample data to IBM COS for testing the pipeline.
#
# Prerequisites:
#   - ibmcloud CLI with cos plugin, or aws CLI configured for COS
#   - .env file with COS credentials
#
# Usage:
#   ./scripts/upload_sample_data.sh <path-to-parquet-file>

set -euo pipefail

source .env

FILE="${1:?Usage: $0 <path-to-parquet-file>}"
DATE=$(date -u +%Y_%m_%d)
KEY="raw/electricity_prices_${DATE}.parquet"

echo "Uploading ${FILE} → cos://${COS_BUCKET}/${KEY}"

# Using ibmcloud cos CLI
ibmcloud cos put-object \
    --bucket "${COS_BUCKET}" \
    --key "${KEY}" \
    --body "${FILE}" \
    --region eu-de

echo "✓ Upload complete: ${KEY}"

