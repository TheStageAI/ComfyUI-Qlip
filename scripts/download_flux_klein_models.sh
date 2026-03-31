#!/bin/bash

# Download Models for FLUX.2 Klein
# This script downloads all necessary models from HuggingFace

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FLUX.2 Klein Model Download Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_PATH="${COMFYUI_PATH:-$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/cache}"

# Check if ComfyUI path exists
if [ ! -d "$COMFYUI_PATH" ]; then
    echo -e "${YELLOW}ComfyUI not found at: $COMFYUI_PATH${NC}"
    echo -e "${YELLOW}Please set COMFYUI_PATH environment variable${NC}"
    echo ""
    echo "Example:"
    echo "  export COMFYUI_PATH=/path/to/ComfyUI"
    echo "  bash download_flux_klein_models.sh"
    exit 1
fi

echo -e "${GREEN}ComfyUI Path: $COMFYUI_PATH${NC}"
echo ""

# Create directories
echo -e "${YELLOW}Creating model directories...${NC}"
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"
echo -e "${GREEN}Directories created${NC}"
echo ""

# Ensure huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}huggingface_hub not found. Installing...${NC}"
    python3 -m pip install "huggingface_hub[cli]"
fi

# Get HF token (required — FLUX.2 Klein is a gated model)
HF_TOKEN="${HF_TOKEN:-$(python3 -c "from huggingface_hub import HfFolder; print(HfFolder.get_token() or '')" 2>/dev/null)}"
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}HuggingFace token not found.${NC}"
    echo -e "${YELLOW}Please login first:${NC}"
    echo "  huggingface-cli login"
    echo -e "${YELLOW}Or set HF_TOKEN:${NC}"
    echo "  export HF_TOKEN=hf_xxxxx"
    echo ""
    echo -e "${YELLOW}Also make sure you accepted the license at:${NC}"
    echo "  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
    echo "  https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B"
    exit 1
fi

# Function to download file from a direct URL with auth
download_url() {
    local url=$1
    local output_dir=$2
    local description=$3
    local filename
    filename=$(basename "$url")

    echo -e "${YELLOW}Downloading: $description${NC}"
    echo -e "  URL: $url"
    echo -e "  To: $output_dir/$filename"

    mkdir -p "$output_dir"

    if [ -f "$output_dir/$filename" ]; then
        echo -e "${GREEN}Already exists: $output_dir/$filename${NC}"
        echo ""
        return 0
    fi

    if command -v wget &> /dev/null; then
        wget -c --header="Authorization: Bearer $HF_TOKEN" -O "$output_dir/$filename" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -C - -H "Authorization: Bearer $HF_TOKEN" -o "$output_dir/$filename" "$url"
    else
        echo -e "${RED}Neither wget nor curl found. Please install one of them.${NC}"
        return 1
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Downloaded: $description${NC}"
    else
        echo -e "${RED}Failed to download: $description${NC}"
        return 1
    fi
    echo ""
}

# Function to download a single file from HF repo via Python API
download_hf() {
    local repo=$1
    local file_path=$2
    local output_dir=$3
    local description=$4

    echo -e "${YELLOW}Downloading: $description${NC}"
    echo -e "  From: $repo/$file_path"
    echo -e "  To: $output_dir"

    mkdir -p "$output_dir"

    local dest="$output_dir/$(basename "$file_path")"
    if [ -f "$dest" ]; then
        echo -e "${GREEN}Already exists: $dest${NC}"
        echo ""
        return 0
    fi

    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$repo', filename='$file_path', token='$HF_TOKEN')
dest = os.path.join('$output_dir', os.path.basename('$file_path'))
shutil.copy2(path, dest)
print(f'Saved to: {dest}')
"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Downloaded: $description${NC}"
    else
        echo -e "${RED}Failed to download: $description${NC}"
        return 1
    fi
    echo ""
}

# ============================================================
# 1. Diffusion Models (BF16 — required for Qlip compilation)
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}1. Diffusion Models (BF16)${NC}"
echo -e "${GREEN}========================================${NC}"

download_hf "black-forest-labs/FLUX.2-klein-9B" \
    "flux-2-klein-9b.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "FLUX.2 Klein 9B BF16"

download_hf "black-forest-labs/FLUX.2-klein-base-9B" \
    "flux-2-klein-base-9b.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "FLUX.2 Klein Base 9B BF16"

# ============================================================
# 2. Text Encoders
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}2. Text Encoder${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "Qwen 3 8B FP8 Mixed Text Encoder"

# ============================================================
# 3. VAE
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}3. VAE${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "$COMFYUI_PATH/models/vae" \
    "FLUX.2 VAE"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Models downloaded to:"
echo -e "  Diffusion Models: ${YELLOW}$COMFYUI_PATH/models/diffusion_models/${NC}"
echo -e "  Text Encoder:     ${YELLOW}$COMFYUI_PATH/models/text_encoders/${NC}"
echo -e "  VAE:              ${YELLOW}$COMFYUI_PATH/models/vae/${NC}"
echo ""
echo -e "${GREEN}Model Storage Layout:${NC}"
echo ""
echo "  ComfyUI/"
echo "  models/"
echo "    diffusion_models/"
echo "      flux-2-klein-9b.safetensors          (BF16)"
echo "      flux-2-klein-base-9b.safetensors      (BF16)"
echo "    text_encoders/"
echo "      qwen_3_8b_fp8mixed.safetensors"
echo "    vae/"
echo "      flux2-vae.safetensors"
echo ""

# Check disk space
echo "Disk space used by models:"
du -sh "$COMFYUI_PATH/models/diffusion_models" 2>/dev/null || echo "  Diffusion models: N/A"
du -sh "$COMFYUI_PATH/models/text_encoders" 2>/dev/null || echo "  Text encoders: N/A"
du -sh "$COMFYUI_PATH/models/vae" 2>/dev/null || echo "  VAE: N/A"
echo ""
