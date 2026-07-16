#!/usr/bin/env bash
# mirror_models.sh — download all Rechroma model weights for self-hosting.
# Usage: ./mirror_models.sh [target_dir]   (default: ./rechroma-models)
# Produces the weights + SHA256SUMS manifest. Upload the folder to your own
# mirror (GitHub release, MinIO, nginx) and point Rechroma at it via
# MODEL_BASE_URL.
set -euo pipefail

DEST="${1:-./rechroma-models}"
mkdir -p "$DEST"
cd "$DEST"

URLS=(
  # DeOldify (MIT)
  "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
  "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth"
  # DDColor (Apache-2.0)
  "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth"
  "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth"
  # GFPGAN + face helpers (Apache-2.0)
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/detection_Resnet50_Final.pth"
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/parsing_parsenet.pth"
  # Real-ESRGAN (BSD-3)
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/realesr-general-x4v3.pth"
)

for url in "${URLS[@]}"; do
  file="${url##*/}"
  if [[ -f "$file" ]]; then
    echo "== $file already present, skipping"
  else
    echo "== downloading $file"
    curl -fL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
done

echo "== generating SHA256SUMS"
sha256sum ./*.pth | tee SHA256SUMS

# Sanity-check the two known-good checksums
grep -q "3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477  ./ColorizeArtistic_gen.pth" SHA256SUMS \
  && echo "OK ColorizeArtistic_gen.pth checksum matches" \
  || echo "WARNING: ColorizeArtistic_gen.pth checksum mismatch!"
grep -q "ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af  ./ColorizeStable_gen.pth" SHA256SUMS \
  && echo "OK ColorizeStable_gen.pth checksum matches" \
  || echo "WARNING: ColorizeStable_gen.pth checksum mismatch!"

echo
echo "Done. Total size:"
du -sh .
echo "Upload this folder to your mirror, then set MODEL_BASE_URL to its base URL."
echo "Copy the SHA256SUMS values into app/core/model_registry.py."
