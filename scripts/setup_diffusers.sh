#!/usr/bin/env bash
# Fetch the diffusers this repo needs: upstream huggingface/diffusers at the pinned commit
# plus the patches in diffusers_patches/, then install it editable.
#
#   bash scripts/setup_diffusers.sh            # -> ./diffusers, installed into the active env
#   DIFFUSERS_DIR=/elsewhere bash scripts/setup_diffusers.sh
#
# Re-running on an existing checkout is safe: it resets to the pin and re-applies.
set -euo pipefail
cd "$(dirname "$0")/.."

PIN=$(sed -n 's/^upstream base: \([0-9a-f]*\).*/\1/p' diffusers_patches/BASE.txt)
DIR=${DIFFUSERS_DIR:-diffusers}
PIP=${PIP:-uv pip}          # PIP="pip" if you do not use uv

if [ ! -d "$DIR/.git" ]; then
  git clone --quiet https://github.com/huggingface/diffusers.git "$DIR"
fi

git -C "$DIR" fetch --quiet origin "$PIN" 2>/dev/null || git -C "$DIR" fetch --quiet origin
git -C "$DIR" checkout --quiet --detach "$PIN"
git -C "$DIR" am --quiet "$PWD"/diffusers_patches/*.patch
echo "diffusers: $PIN + $(ls diffusers_patches/*.patch | wc -l) patches -> $(git -C "$DIR" rev-parse --short HEAD)"

$PIP install -e "$DIR"
