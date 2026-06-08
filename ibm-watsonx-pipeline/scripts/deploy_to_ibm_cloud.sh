#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# IBM watsonx Pipeline — Full Deployment to IBM Cloud
#
# This script deploys the pipeline to IBM Cloud with REAL credentials.
# Run each section step-by-step (don't just execute the whole file blindly).
#
# Prerequisites:
#   1. IBM Cloud account (https://cloud.ibm.com)
#   2. ibmcloud CLI installed (https://cloud.ibm.com/docs/cli)
#   3. Code Engine plugin: ibmcloud plugin install code-engine
#   4. A watsonx.ai instance provisioned
#   5. A Cloud Object Storage instance provisioned
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "  IBM watsonx Pipeline — Deployment Guide"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── STEP 1: IBM Cloud Login ────────────────────────────────────────────────
echo "═══ STEP 1: Login to IBM Cloud ═══"
echo ""
echo "  Run interactively:"
echo "    ibmcloud login"
echo ""
echo "  Or with API key:"
echo "    ibmcloud login --apikey \$IBMCLOUD_API_KEY -r eu-de -g Default"
echo ""
read -p "Press Enter after logging in..."

# ─── STEP 2: Create Cloud Object Storage Bucket ────────────────────────────
echo ""
echo "═══ STEP 2: Verify Cloud Object Storage ═══"
echo ""
echo "  You need:"
echo "    - A COS instance (check: ibmcloud resource service-instances --service-name cloud-object-storage)"
echo "    - A bucket named 'watsonx-pipeline-bucket'"
echo ""
echo "  Create via UI: https://cloud.ibm.com/objectstorage"
echo "  Or CLI:"
echo "    ibmcloud cos bucket-create --bucket watsonx-pipeline-bucket --region eu-de"
echo ""
echo "  Get your COS credentials:"
echo "    ibmcloud resource service-keys --instance-name <your-cos-instance>"
echo ""
read -p "Press Enter when COS is ready..."

# ─── STEP 3: Get watsonx.ai Credentials ───────────────────────────────────
echo ""
echo "═══ STEP 3: Get watsonx.ai Credentials ═══"
echo ""
echo "  You need from https://dataplatform.cloud.ibm.com :"
echo ""
echo "    WATSONX_API_KEY     → IBM Cloud > Manage > Access (IAM) > API Keys > Create"
echo "    WATSONX_PROJECT_ID  → watsonx.ai > Your project > Manage > General > Project ID"
echo "    WATSONX_URL         → https://eu-de.ml.cloud.ibm.com (Frankfurt)"
echo "                          https://us-south.ml.cloud.ibm.com (Dallas)"
echo "    WATSONX_SPACE_ID    → watsonx.ai > Deployments > Spaces > Your space > Settings > Space ID"
echo ""
echo "  For the deployment space, create one at:"
echo "    https://dataplatform.cloud.ibm.com/ml-runtime/spaces"
echo ""
read -p "Press Enter when you have your credentials..."

# ─── STEP 4: Create .env file ─────────────────────────────────────────────
echo ""
echo "═══ STEP 4: Create .env file ═══"
echo ""
echo "  Copy .env.example and fill in your credentials:"
echo ""
echo "    cp .env.example .env"
echo "    # Edit .env with your real values"
echo ""

if [ ! -f .env ]; then
    echo "  ⚠️  .env file not found!"
    echo "  Creating from template..."
    cp .env.example .env
    echo "  → Please edit .env with your real credentials now."
    echo ""
    read -p "Press Enter after editing .env..."
else
    echo "  ✓ .env file exists"
fi

# ─── STEP 5: Upload sample data to COS ────────────────────────────────────
echo ""
echo "═══ STEP 5: Upload sample data to COS ═══"
echo ""
echo "  This generates synthetic data and uploads to your COS bucket."
echo ""
read -p "Run data upload? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python -c "
from scripts.prepare_local_data import generate_sample_data
from src.storage.cos_client import COSClient
from src.pipeline.config import PipelineConfig

config = PipelineConfig.from_env()
cos = COSClient(config)

df = generate_sample_data(days=90)
key = 'raw/electricity_prices_2026_06_01.parquet'
cos.write_parquet(df, key)
print(f'✓ Uploaded {len(df)} rows to cos://{config.cos_bucket}/{key}')
"
fi

# ─── STEP 6: Test pipeline with real COS (no watsonx deploy) ──────────────
echo ""
echo "═══ STEP 6: Test Training Pipeline with real COS ═══"
echo ""
echo "  This runs training using real COS but mock watsonx (to verify COS works):"
echo ""
echo "    uv run python -m src.pipeline.orchestrator --mode training --local"
echo ""
echo "  To run with REAL watsonx.ai (registers model in your project):"
echo ""
echo "    uv run python -m src.pipeline.orchestrator --mode training"
echo ""
read -p "Run training with real COS + real watsonx? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    uv run python -m src.pipeline.orchestrator --mode training
fi

# ─── STEP 7: Deploy to Code Engine ────────────────────────────────────────
echo ""
echo "═══ STEP 7: Deploy to IBM Cloud Code Engine ═══"
echo ""
echo "  This containerizes the pipeline and deploys as a scheduled job."
echo ""
read -p "Deploy to Code Engine? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Create/select project
    ibmcloud ce project select --name watsonx-pipeline 2>/dev/null || \
        ibmcloud ce project create --name watsonx-pipeline

    # Build image
    echo "  Building container image..."
    ibmcloud ce buildrun submit \
        --name watsonx-build-$(date +%s) \
        --source . \
        --strategy dockerfile \
        --image private.de.icr.io/watsonx-ml/watsonx-pipeline:latest \
        --registry-secret icr-secret \
        --wait

    # Create/update secrets
    echo "  Creating secrets..."
    ibmcloud ce secret update --name watsonx-credentials --from-env-file .env 2>/dev/null || \
        ibmcloud ce secret create --name watsonx-credentials --from-env-file .env

    # Create training job
    echo "  Creating training job..."
    ibmcloud ce job create --name watsonx-training \
        --image private.de.icr.io/watsonx-ml/watsonx-pipeline:latest \
        --registry-secret icr-secret \
        --env-from-secret watsonx-credentials \
        --argument "--mode" --argument "training" \
        --cpu 4 --memory 8G \
        --maxexecutiontime 3600 \
        2>/dev/null || \
    ibmcloud ce job update --name watsonx-training \
        --image private.de.icr.io/watsonx-ml/watsonx-pipeline:latest

    # Create inference job
    echo "  Creating inference job..."
    ibmcloud ce job create --name watsonx-inference \
        --image private.de.icr.io/watsonx-ml/watsonx-pipeline:latest \
        --registry-secret icr-secret \
        --env-from-secret watsonx-credentials \
        --argument "--mode" --argument "inference" \
        --cpu 2 --memory 4G \
        --maxexecutiontime 1800 \
        2>/dev/null || \
    ibmcloud ce job update --name watsonx-inference \
        --image private.de.icr.io/watsonx-ml/watsonx-pipeline:latest

    # Schedule
    echo "  Setting up schedules..."
    ibmcloud ce subscription cron create --name weekly-watsonx-training \
        --destination-type job --destination watsonx-training \
        --schedule "0 6 * * 0" 2>/dev/null || echo "  (already exists)"

    ibmcloud ce subscription cron create --name daily-watsonx-inference \
        --destination-type job --destination watsonx-inference \
        --schedule "0 8 * * *" 2>/dev/null || echo "  (already exists)"

    echo ""
    echo "  ✓ Deployed to Code Engine!"
fi

# ─── STEP 8: Trigger a manual run ─────────────────────────────────────────
echo ""
echo "═══ STEP 8: Trigger Manual Run ═══"
echo ""
echo "  Run training now:"
echo "    ibmcloud ce jobrun submit --job watsonx-training"
echo ""
echo "  Run inference now:"
echo "    ibmcloud ce jobrun submit --job watsonx-inference"
echo ""
echo "  View logs:"
echo "    ibmcloud ce jobrun list"
echo "    ibmcloud ce jobrun logs -f -n <run-name>"
echo ""

# ─── STEP 9: Verify in watsonx.ai UI ──────────────────────────────────────
echo ""
echo "═══ STEP 9: Verify in watsonx.ai UI ═══"
echo ""
echo "  After training runs, check:"
echo ""
echo "  1. Model Registry:"
echo "     https://dataplatform.cloud.ibm.com → Your Project → Assets → Models"
echo "     You should see 'watsonx-energy-forecast-model'"
echo ""
echo "  2. Deployments:"
echo "     https://dataplatform.cloud.ibm.com → Deployments → Your Space"
echo "     You should see 'energy-forecast-deployment' with a REST endpoint"
echo ""
echo "  3. COS Bucket:"
echo "     https://cloud.ibm.com/objectstorage → watsonx-pipeline-bucket"
echo "     Check: raw/, models/, features/, forecasts/, pipeline-metadata/dag/"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  ✓ Deployment guide complete!"
echo "═══════════════════════════════════════════════════════════"

