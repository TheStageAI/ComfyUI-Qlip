#!/bin/bash
set -e

COMFYUI_PATH="${COMFYUI_PATH:-./ComfyUI}"
HF_TOKEN="${HF_TOKEN:-$(python3 -c "from huggingface_hub import HfFolder; print(HfFolder.get_token() or '')" 2>/dev/null)}"

mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"

download_url() {
    local url=$1 output_dir=$2 description=$3
    local filename=$(basename "$url")
    echo "Downloading: $description"
    if [ -f "$output_dir/$filename" ]; then
        echo "  Already exists, skipping"
        return 0
    fi
    curl -L -C - -H "Authorization: Bearer $HF_TOKEN" -o "$output_dir/$filename" "$url"
    echo "  Done: $output_dir/$filename"
}

# Diffusion model (BF16 — full precision for Qlip to quantize itself)
download_url \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors" \
    "$COMFYUI_PATH/models/diffusion_models" \
    "Qwen Image Edit diffusion model (BF16, ~40.9 GB)"

# Text encoder (Qwen 2.5 VL 7B FP8)
download_url \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
    "$COMFYUI_PATH/models/text_encoders" \
    "Qwen 2.5 VL 7B text encoder (FP8)"

# VAE
download_url \
    "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors" \
    "$COMFYUI_PATH/models/vae" \
    "Qwen Image VAE"

# LoRA — Lightning 4-step acceleration
download_url \
    "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/loras/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \
    "$COMFYUI_PATH/models/loras" \
    "Qwen Image Edit Lightning 4-step LoRA"

echo ""
echo "All models downloaded to $COMFYUI_PATH/models/"
echo ""
echo "Files:"
echo "  diffusion_models/qwen_image_bf16.safetensors"
echo "  text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
echo "  vae/qwen_image_vae.safetensors"
echo "  loras/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
