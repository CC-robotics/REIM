#!/usr/bin/env bash
# Compile the current IEEE manuscript and publish the named submission PDF.

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_DIR="$PROJECT_DIR/manuscript"
SOURCE="$SOURCE_DIR/main.tex"
BUILD_PDF="$SOURCE_DIR/main.pdf"
OUTPUT_DIR="$PROJECT_DIR/output/pdf"
OUTPUT="$OUTPUT_DIR/REIM_main_four_seed.pdf"
TECTONIC_VERSION="0.16.9"
TECTONIC_SHA256="60b13a0826ae7ad9ce34b4a2df06bff2cfcfa6dda8a915477c0cbb84e1a4a902"
LOCAL_ENGINE="$PROJECT_DIR/.tools/tectonic/tectonic"
LOCAL_ENGINE_WINDOWS="$PROJECT_DIR/.tools/tectonic/tectonic.exe"

[[ -f "$SOURCE" ]] || {
  printf 'Missing %s; generate paper assets first.\n' "$SOURCE" >&2
  exit 2
}

if command -v tectonic >/dev/null 2>&1; then
  ENGINE="$(command -v tectonic)"
elif [[ -x "$LOCAL_ENGINE" ]]; then
  ENGINE="$LOCAL_ENGINE"
elif [[ -x "$LOCAL_ENGINE_WINDOWS" ]]; then
  ENGINE="$LOCAL_ENGINE_WINDOWS"
else
  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
    printf 'Install Tectonic manually for this platform.\n' >&2
    exit 2
  }
  DOWNLOAD_DIR="$(mktemp -d)"
  trap 'rm -rf -- "$DOWNLOAD_DIR"' EXIT
  ARCHIVE="$DOWNLOAD_DIR/tectonic.tar.gz"
  URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz"
  printf 'Installing Tectonic %s locally...\n' "$TECTONIC_VERSION"
  curl -fL --retry 3 "$URL" -o "$ARCHIVE"
  printf '%s  %s\n' "$TECTONIC_SHA256" "$ARCHIVE" | sha256sum --check -
  mkdir -p "$(dirname -- "$LOCAL_ENGINE")"
  tar -xzf "$ARCHIVE" -C "$(dirname -- "$LOCAL_ENGINE")"
  ENGINE="$LOCAL_ENGINE"
fi

(
  cd "$SOURCE_DIR"
  "$ENGINE" --keep-logs main.tex
)
[[ -s "$BUILD_PDF" ]] || {
  printf 'LaTeX compilation did not produce %s\n' "$BUILD_PDF" >&2
  exit 1
}
mkdir -p "$OUTPUT_DIR"
cp "$BUILD_PDF" "$OUTPUT"
printf 'Compiled %s\n' "$OUTPUT"
