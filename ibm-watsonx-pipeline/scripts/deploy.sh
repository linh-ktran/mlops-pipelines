#!/bin/bash
# Deploy the watsonx pipeline to IBM Cloud Code Engine
# Usage: ./scripts/deploy.sh

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_NAME="watsonx-pipeline"
IMAGE_NAME="de.icr.io/energy-ml/watsonx-pipeline:latest"
REGION="eu-de"
RESOURCE_GROUP="Test"

echo "═══════════════════════════════════════════════════════"
echo "  IBM watsonx Pipeline — Deployment"
echo "═══════════════════════════════════════════════════════"
echo ""

# ─── Load API key from .env if not already set ───────────────────────────────
if [ -z "${IBMCLOUD_API_KEY:-}" ]; then
    if [ -f .env ]; then
        export IBMCLOUD_API_KEY=$(grep '^WATSONX_API_KEY=' .env | cut -d '=' -f2- | tr -d '"' | tr -d "'")
        echo "  (loaded IBMCLOUD_API_KEY from .env)"
    fi
fi

# ─── 1. Login ────────────────────────────────────────────────────────────────
echo "→ Step 1: IBM Cloud Login"
if ibmcloud target 2>/dev/null | grep -q "khanh-linh"; then
    echo "  (already logged in, re-targeting region/group)"
    ibmcloud target -r "$REGION" -g "$RESOURCE_GROUP"
elif [ -n "${IBMCLOUD_API_KEY:-}" ]; then
    ibmcloud login --apikey "$IBMCLOUD_API_KEY" -r "$REGION" -g "$RESOURCE_GROUP"
else
    echo "  No API key found. Logging in interactively..."
    ibmcloud login -r "$REGION" -g "$RESOURCE_GROUP"
fi
echo ""

# ─── 2. Select Code Engine project ──────────────────────────────────────────
echo "→ Step 2: Select Code Engine project"
ibmcloud ce project select --name "$PROJECT_NAME" 2>/dev/null || \
    ibmcloud ce project create --name "$PROJECT_NAME"
echo ""

# ─── 3. Build container image ───────────────────────────────────────────────
echo "→ Step 3: Build container image"
ibmcloud ce buildrun submit \
    --name watsonx-pipeline-build \
    --source . \
    --strategy dockerfile \
    --image "$IMAGE_NAME" \
    --registry-secret icr-secret \
    --wait
echo ""

# ─── 4. Create secrets ──────────────────────────────────────────────────────
echo "→ Step 4: Create/update secrets"
ibmcloud ce secret update --name watsonx-credentials --from-env-file .env 2>/dev/null || \
    ibmcloud ce secret create --name watsonx-credentials --from-env-file .env
echo ""

# ─── 5. Create training job ─────────────────────────────────────────────────
echo "→ Step 5: Create training job"
ibmcloud ce job create --name watsonx-training \
    --image "$IMAGE_NAME" \
    --registry-secret icr-secret \
    --env-from-secret watsonx-credentials \
    --argument "--mode" --argument "training" \
    --cpu 4 --memory 8G \
    --maxexecutiontime 3600 \
    2>/dev/null || \
ibmcloud ce job update --name watsonx-training \
    --image "$IMAGE_NAME" \
    --argument "--mode" --argument "training"
echo ""

# ─── 6. Create inference job ────────────────────────────────────────────────
echo "→ Step 6: Create inference job"
ibmcloud ce job create --name watsonx-inference \
    --image "$IMAGE_NAME" \
    --registry-secret icr-secret \
    --env-from-secret watsonx-credentials \
    --argument "--mode" --argument "inference" \
    --cpu 2 --memory 4G \
    --maxexecutiontime 1800 \
    2>/dev/null || \
ibmcloud ce job update --name watsonx-inference \
    --image "$IMAGE_NAME" \
    --argument "--mode" --argument "inference"
echo ""

# ─── 7. Schedule jobs ───────────────────────────────────────────────────────
echo "→ Step 7: Schedule cron jobs"

# Daily inference at 08:00 UTC
ibmcloud ce subscription cron create --name daily-watsonx-inference \
    --destination-type job \
    --destination watsonx-inference \
    --schedule "0 8 * * *" \
    2>/dev/null || echo "  (subscription already exists)"

# Weekly training on Sunday at 06:00 UTC
ibmcloud ce subscription cron create --name weekly-watsonx-training \
    --destination-type job \
    --destination watsonx-training \
    --schedule "0 6 * * 0" \
    2>/dev/null || echo "  (subscription already exists)"
echo ""

# ─── Done ────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  Manual run:"
echo "    ibmcloud ce jobrun submit --job watsonx-training"
echo "    ibmcloud ce jobrun submit --job watsonx-inference"
echo ""
echo "  View logs:"
echo "    ibmcloud ce jobrun logs -f -n <run-name>"
echo "═══════════════════════════════════════════════════════"
