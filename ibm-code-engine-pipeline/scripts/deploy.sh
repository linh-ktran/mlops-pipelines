#!/usr/bin/env bash
# Deploy the energy forecast pipeline to IBM Cloud Code Engine.
#
# Prerequisites:
#   - ibmcloud CLI installed with Code Engine plugin
#   - Logged in: ibmcloud login
#   - Container Registry namespace created
#
# Usage:
#   ./scripts/deploy.sh

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_NAME="${CE_PROJECT:-energy-forecast}"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-icr.io}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-energy-ml}"
IMAGE_NAME="energy-forecast-pipeline"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "═══════════════════════════════════════════════════════"
echo "  IBM Cloud Code Engine — Energy Forecast Deployment"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Project:  ${PROJECT_NAME}"
echo "  Image:    ${FULL_IMAGE}"
echo ""

# ─── Select Code Engine project ─────────────────────────────────────────────
echo "→ Selecting Code Engine project: ${PROJECT_NAME}"
ibmcloud ce project select --name "${PROJECT_NAME}" 2>/dev/null || {
    echo "→ Project not found. Creating: ${PROJECT_NAME}"
    ibmcloud ce project create --name "${PROJECT_NAME}"
    ibmcloud ce project select --name "${PROJECT_NAME}"
}

# ─── Build container image ───────────────────────────────────────────────────
echo ""
echo "→ Building container image..."
ibmcloud ce buildrun submit \
    --name "forecast-build-$(date +%s)" \
    --source . \
    --strategy dockerfile \
    --image "${FULL_IMAGE}" \
    --wait

# ─── Create secrets (if not exist) ──────────────────────────────────────────
echo ""
echo "→ Creating/updating COS credentials secret..."
ibmcloud ce secret update --name cos-credentials \
    --from-env-file .env 2>/dev/null || \
ibmcloud ce secret create --name cos-credentials \
    --from-env-file .env

# ─── Create/update inference job (daily) ─────────────────────────────────────
echo ""
echo "→ Creating inference job (daily at 08:00 UTC)..."
ibmcloud ce job update --name forecast-inference \
    --image "${FULL_IMAGE}" \
    --env-from-secret cos-credentials \
    --arg "--mode" --arg "inference" \
    --cpu 2 --memory 4G \
    --maxexecutiontime 1800 2>/dev/null || \
ibmcloud ce job create --name forecast-inference \
    --image "${FULL_IMAGE}" \
    --env-from-secret cos-credentials \
    --arg "--mode" --arg "inference" \
    --cpu 2 --memory 4G \
    --maxexecutiontime 1800

# Create daily cron schedule
ibmcloud ce subscription cron update --name daily-inference \
    --destination-type job \
    --destination forecast-inference \
    --schedule "0 8 * * *" 2>/dev/null || \
ibmcloud ce subscription cron create --name daily-inference \
    --destination-type job \
    --destination forecast-inference \
    --schedule "0 8 * * *"

# ─── Create/update training job (weekly) ─────────────────────────────────────
echo ""
echo "→ Creating training job (weekly on Sunday at 06:00 UTC)..."
ibmcloud ce job update --name forecast-training \
    --image "${FULL_IMAGE}" \
    --env-from-secret cos-credentials \
    --arg "--mode" --arg "training" \
    --cpu 4 --memory 8G \
    --maxexecutiontime 3600 2>/dev/null || \
ibmcloud ce job create --name forecast-training \
    --image "${FULL_IMAGE}" \
    --env-from-secret cos-credentials \
    --arg "--mode" --arg "training" \
    --cpu 4 --memory 8G \
    --maxexecutiontime 3600

# Create weekly cron schedule
ibmcloud ce subscription cron update --name weekly-training \
    --destination-type job \
    --destination forecast-training \
    --schedule "0 6 * * 0" 2>/dev/null || \
ibmcloud ce subscription cron create --name weekly-training \
    --destination-type job \
    --destination forecast-training \
    --schedule "0 6 * * 0"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  Daily inference:  08:00 UTC every day"
echo "  Weekly training:  06:00 UTC every Sunday"
echo ""
echo "  Manual run:"
echo "    ibmcloud ce jobrun submit --job forecast-inference"
echo "    ibmcloud ce jobrun submit --job forecast-training"
echo ""
echo "  Monitor:"
echo "    ibmcloud ce jobrun list"
echo "    ibmcloud ce jobrun logs --name <run-name>"
echo "═══════════════════════════════════════════════════════"

