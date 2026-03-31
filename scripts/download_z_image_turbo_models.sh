#!/bin/bash

# Download Models for Z-Image-Turbo
# This script downloads all necessary models from HuggingFace

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Z-Image-Turbo Model Download Script${NC}"
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
    echo "  bash download_z_image_turbo_models.sh"
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
# 1. Diffusion Model
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}1. Diffusion Model${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "Z-Image-Turbo BF16"

# ============================================================
# 2. Text Encoder
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}2. Text Encoder (Qwen 3 4B)${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "Qwen 3 4B Text Encoder"

# ============================================================
# 3. VAE
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}3. VAE${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
    "$COMFYUI_PATH/models/vae" \
    "Z-Image-Turbo VAE (ae)"

# ============================================================
# 4. LoRAs (optional)
# ============================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}4. LoRAs (optional)${NC}"
echo -e "${GREEN}========================================${NC}"

download_url \
    "https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "Pixel Art Style LoRA for Z-Image-Turbo"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Models downloaded to:"
echo -e "  Diffusion Model:  ${YELLOW}$COMFYUI_PATH/models/diffusion_models/${NC}"
echo -e "  Text Encoder:     ${YELLOW}$COMFYUI_PATH/models/text_encoders/${NC}"
echo -e "  VAE:              ${YELLOW}$COMFYUI_PATH/models/vae/${NC}"
echo -e "  LoRAs:            ${YELLOW}$COMFYUI_PATH/models/loras/${NC}"
echo ""
echo -e "${GREEN}Model Storage Layout:${NC}"
echo ""
echo "  ComfyUI/"
echo "  models/"
echo "    diffusion_models/"
echo "      z_image_turbo_bf16.safetensors"
echo "    text_encoders/"
echo "      qwen_3_4b.safetensors"
echo "    vae/"
echo "      ae.safetensors"
echo "    loras/"
echo "      pixel_art_style_z_image_turbo.safetensors"
echo ""

# Check disk space
echo "Disk space used by models:"
du -sh "$COMFYUI_PATH/models/diffusion_models" 2>/dev/null || echo "  Diffusion models: N/A"
du -sh "$COMFYUI_PATH/models/text_encoders" 2>/dev/null || echo "  Text encoders: N/A"
du -sh "$COMFYUI_PATH/models/vae" 2>/dev/null || echo "  VAE: N/A"
du -sh "$COMFYUI_PATH/models/loras" 2>/dev/null || echo "  LoRAs: N/A"
echo ""
