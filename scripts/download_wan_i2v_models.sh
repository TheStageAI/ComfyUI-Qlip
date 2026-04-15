#!/bin/bash
# Download Models for Wan2.2 I2V (Image-to-Video) 14B
# Two transformers: high_noise + low_noise

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================"
echo -e "Wan2.2 I2V (14B) Model Download"
echo -e "========================================${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_PATH="${COMFYUI_PATH:-$SCRIPT_DIR/ComfyUI}"

if [ ! -d "$COMFYUI_PATH" ]; then
    echo -e "${YELLOW}ComfyUI not found at: $COMFYUI_PATH${NC}"
    echo "Set COMFYUI_PATH and re-run."
    exit 1
fi

echo -e "${GREEN}ComfyUI: $COMFYUI_PATH${NC}"
echo ""

mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/clip_vision"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    pip3 install "huggingface_hub[cli]"
fi

download_url() {
    local url=$1
    local output_dir=$2
    local description=$3
    local filename
    filename=$(basename "$url")

    echo -e "${YELLOW}Downloading: $description${NC}"
    echo "  URL: $url"
    echo "  To: $output_dir/$filename"

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
        echo -e "${RED}Neither wget nor curl found.${NC}"
        return 1
    fi
    echo -e "${GREEN}Downloaded: $description${NC}"
    echo ""
}

# ============================================================
# 1. Diffusion Models (BF16 — two transformers)
# ============================================================
echo -e "${GREEN}1. Diffusion Models (high_noise + low_noise)${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "Wan2.2 I2V High Noise 14B (FP16)"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "Wan2.2 I2V Low Noise 14B (FP16)"

# ============================================================
# 2. Text Encoder (UMT5-XXL)
# ============================================================
echo -e "${GREEN}2. Text Encoder${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "UMT5-XXL FP8 Text Encoder"

# ============================================================
# 3. CLIP Vision (for I2V)
# ============================================================
echo -e "${GREEN}3. CLIP Vision${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "$COMFYUI_PATH/models/clip_vision" \
    "CLIP Vision H"

# ============================================================
# 4. VAE
# ============================================================
echo -e "${GREEN}4. VAE${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$COMFYUI_PATH/models/vae" \
    "Wan 2.1 VAE"

# ============================================================
# 5. LoRAs (4-step distill — one per transformer)
# ============================================================
echo -e "${GREEN}5. LoRAs (4-step distill)${NC}"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "Wan2.2 I2V 4-step LoRA (high noise)"

download_url \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "Wan2.2 I2V 4-step LoRA (low noise)"

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${GREEN}Download complete!${NC}"
echo ""
echo "  ComfyUI/models/"
echo "    diffusion_models/"
echo "      wan2.2_i2v_high_noise_14B_fp16.safetensors"
echo "      wan2.2_i2v_low_noise_14B_fp16.safetensors"
echo "    text_encoders/"
echo "      umt5_xxl_fp8_e4m3fn_scaled.safetensors"
echo "    clip_vision/"
echo "      clip_vision_h.safetensors"
echo "    vae/"
echo "      wan_2.1_vae.safetensors"
echo "    loras/"
echo "      wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
echo "      wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
