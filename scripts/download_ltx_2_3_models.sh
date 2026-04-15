#!/bin/bash

# Download Models for LTX-2.3 Video (22B)
# This script downloads all necessary models from HuggingFace

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LTX-2.3 (22B) Model Download Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_PATH="${COMFYUI_PATH:-$SCRIPT_DIR/ComfyUI}"

# Check if ComfyUI path exists
if [ ! -d "$COMFYUI_PATH" ]; then
    echo -e "${YELLOW}ComfyUI not found at: $COMFYUI_PATH${NC}"
    echo -e "${YELLOW}Please set COMFYUI_PATH environment variable or install ComfyUI first${NC}"
    echo ""
    echo "Example:"
    echo "  export COMFYUI_PATH=/path/to/ComfyUI"
    echo "  bash download_ltx_2_3_models.sh"
    exit 1
fi

echo -e "${GREEN}ComfyUI Path: $COMFYUI_PATH${NC}"
echo ""

# Create directories
echo -e "${YELLOW}Creating model directories...${NC}"
mkdir -p "$COMFYUI_PATH/models/checkpoints"
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/loras"
mkdir -p "$COMFYUI_PATH/models/latent_upscale_models"
echo -e "${GREEN}Directories created${NC}"
echo ""

# Ensure huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}huggingface_hub not found. Installing...${NC}"
    python3 -m pip install "huggingface_hub[cli]"
fi

# Function to download file from a direct URL
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
        wget -c -O "$output_dir/$filename" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$output_dir/$filename" "$url"
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

# ============================================================
# 1. Checkpoints (includes VAE)
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}1. Checkpoints (LTX-2.3 22B)${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors" \
    "$COMFYUI_PATH/models/checkpoints" \
    "LTX-2.3 22B Dev (BF16)"

#download_url \
#    "https://huggingface.co/Lightricks/LTX-Video-2/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
#    "$COMFYUI_PATH/models/checkpoints" \
#    "LTX-2.3 22B Dev (FP8)"

# ============================================================
# 2. Text Encoder (Gemma 3 12B)
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}2. Text Encoder (Gemma 3 12B)${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "Gemma 3 12B IT FP4 Mixed"

# ============================================================
# 3. LoRAs
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}3. LoRAs${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "LTX-2.3 22B Distilled LoRA (rank 384)"

download_url \
    "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/resolve/main/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "LTX-2.3 22B Distilled LoRA (rank 384)"
#download_url \
#    "https://huggingface.co/mlpen/gemma-3-12b-it-abliterated/resolve/main/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors" \
#    "$COMFYUI_PATH/models/loras" \
#    "Gemma 3 12B IT Abliterated LoRA (rank 64)"

# ============================================================
# 4. Latent Upscaler
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}4. Latent Upscaler${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$COMFYUI_PATH/models/latent_upscale_models" \
    "LTX-2.3 Spatial Upscaler x2 v1.1"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Models downloaded to:"
echo -e "  Checkpoints:      ${YELLOW}$COMFYUI_PATH/models/checkpoints/${NC}"
echo -e "  Text Encoder:     ${YELLOW}$COMFYUI_PATH/models/text_encoders/${NC}"
echo -e "  LoRAs:            ${YELLOW}$COMFYUI_PATH/models/loras/${NC}"
echo -e "  Latent Upscaler:  ${YELLOW}$COMFYUI_PATH/models/latent_upscale_models/${NC}"
echo ""
echo -e "${GREEN}Model Storage Layout:${NC}"
echo ""
echo "  ComfyUI/models/"
echo "    checkpoints/"
echo "      ltx-2.3-22b-dev.safetensors"
echo "      ltx-2.3-22b-dev-fp8.safetensors"
echo "    text_encoders/"
echo "      gemma_3_12B_it_fp4_mixed.safetensors"
echo "    loras/"
echo "      ltx-2.3-22b-distilled-lora-384.safetensors"
echo "      gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
echo "    latent_upscale_models/"
echo "      ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
echo ""

# Check disk space
echo "Disk space used by models:"
du -sh "$COMFYUI_PATH/models/checkpoints" 2>/dev/null || echo "  Checkpoints: N/A"
du -sh "$COMFYUI_PATH/models/text_encoders" 2>/dev/null || echo "  Text encoders: N/A"
du -sh "$COMFYUI_PATH/models/loras" 2>/dev/null || echo "  LoRAs: N/A"
du -sh "$COMFYUI_PATH/models/latent_upscale_models" 2>/dev/null || echo "  Upscaler: N/A"
echo ""
