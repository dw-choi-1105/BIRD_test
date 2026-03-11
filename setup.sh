#!/bin/bash
# One-shot setup script for BIRD_test environment
set -e

echo "=== [1/3] Installing Python packages ==="

# PyTorch with CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install -r requirements.txt

echo ""
echo "=== [2/3] Installing VSCode extensions ==="

EXTENSIONS=(
    anthropic.claude-code
    github.copilot-chat
    kisstkondoros.vscode-gutter-preview
    mechatroner.rainbow-csv
    ms-python.debugpy
    ms-python.python
    ms-python.vscode-pylance
    ms-python.vscode-python-envs
)

for ext in "${EXTENSIONS[@]}"; do
    echo "  Installing: $ext"
    code --install-extension "$ext" --force 2>/dev/null || \
        echo "  [skip] $ext (code CLI not available — will auto-install via .vscode/extensions.json on first open)"
done

echo ""
echo "=== [3/3] Downloading model weights (if needed) ==="

if [ ! -f "data/kernel.npy" ]; then
    echo "  Run: python3 download.py  (or place your data/ files manually)"
else
    echo "  data/kernel.npy found — skipping download"
fi

echo ""
echo "Done. Open this folder in VSCode and recommended extensions will be suggested automatically."
