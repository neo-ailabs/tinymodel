#!/bin/sh
# Regenerates the committed SPIR-V binaries in spv/ from shaders/*.comp.
# Usage: TM_GLSLC=/path/to/glslc crates/tm-vulkan/gen_spv.sh
set -eu
cd "$(dirname "$0")"
GLSLC="${TM_GLSLC:-${VULKAN_SDK:+$VULKAN_SDK/bin/}glslc}"
mkdir -p spv
for src in shaders/*.comp; do
    stem=$(basename "$src" .comp)
    "$GLSLC" --target-env=vulkan1.2 -O -I shaders -o "spv/$stem.spv" "$src"
    echo "spv/$stem.spv"
done
