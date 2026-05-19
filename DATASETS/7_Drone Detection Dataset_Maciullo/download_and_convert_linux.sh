#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Linux / RunPod script
# Run from inside 7_DroneDetectionDataset:
#
#   chmod +x download_and_convert_linux.sh
#   ./download_and_convert_linux.sh
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf "$ROOT_DIR/_raw_extracted" "$ROOT_DIR/_raw_downloads" "$ROOT_DIR/test" "$ROOT_DIR/train" "$ROOT_DIR/val"

RAW_DIR="$ROOT_DIR/_raw_downloads"
EXTRACT_DIR="$ROOT_DIR/_raw_extracted"
CONVERTER="$ROOT_DIR/convert_voc_to_yolov26.py"

CLASSES=("drone")

TRAIN_GDRIVE_LINKS=(
  https://drive.google.com/file/d/1lGumpFGdAvLWXhdLJCbkpv-Uo-t8fpHT/view?usp=sharing
)

VAL_GDRIVE_LINKS=(
  https://drive.google.com/file/d/1dJdUIqTfW7InN76Xb3flWVrVmcficP93/view?usp=share_link
)

TEST_GDRIVE_LINKS=(
  # Optional
)

install_requirements() {
  echo "Checking/installing system dependencies..."

  if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip not found. Installing unzip..."

    if command -v apt-get >/dev/null 2>&1; then
      apt-get update
      apt-get install -y unzip
    else
      echo "ERROR: unzip is missing and apt-get is not available."
      echo "Please install unzip manually for this Linux environment."
      exit 1
    fi
  else
    echo "unzip is already installed."
  fi

  echo "Installing Python dependencies..."
  python3 -m pip install --upgrade pip
  python3 -m pip install gdown tqdm
}

make_output_dirs() {
  mkdir -p "$ROOT_DIR/train/images" "$ROOT_DIR/train/labels"
  mkdir -p "$ROOT_DIR/val/images" "$ROOT_DIR/val/labels"
  mkdir -p "$ROOT_DIR/test/images" "$ROOT_DIR/test/labels"
  mkdir -p "$RAW_DIR" "$EXTRACT_DIR"
}

download_gdrive_item() {
  local link="$1"
  local output_dir="$2"

  mkdir -p "$output_dir"

  if [[ -z "$link" || "$link" == "PUT_"* ]]; then
    echo "Skipping placeholder link: $link"
    return 0
  fi

  echo "Downloading: $link"
  gdown "$link" -O "$output_dir/"
}

extract_archives() {
  local input_dir="$1"
  local output_dir="$2"

  mkdir -p "$output_dir"
  shopt -s nullglob

  for file in "$input_dir"/*; do
    echo "Extracting/copying: $file"

    case "$file" in
      *.zip)
        unzip -q "$file" -d "$output_dir"
        ;;
      *.tar.gz|*.tgz)
        tar -xzf "$file" -C "$output_dir"
        ;;
      *.tar)
        tar -xf "$file" -C "$output_dir"
        ;;
      *.rar)
        if command -v unrar >/dev/null 2>&1; then
          unrar x -o+ "$file" "$output_dir/"
        else
          echo "ERROR: .rar file found but unrar is not installed."
          echo "Install it with:"
          echo "apt-get update && apt-get install -y unrar"
          exit 1
        fi
        ;;
      *)
        cp -r "$file" "$output_dir/"
        ;;
    esac
  done

  shopt -u nullglob
}

find_image_dir() {
  local split_extract_dir="$1"

  find "$split_extract_dir" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" -o \
    -iname "*.bmp" \
  \) -printf '%h\n' | sort | uniq | head -n 1
}

find_xml_dir() {
  local split_extract_dir="$1"

  find "$split_extract_dir" -type f -iname "*.xml" -printf '%h\n' | sort | uniq | head -n 1
}

convert_split() {
  local split="$1"
  local split_extract_dir="$2"

  if [[ ! -d "$split_extract_dir" ]]; then
    echo "No extracted folder for split: $split"
    echo "Skipping $split"
    return 0
  fi

  if ! find "$split_extract_dir" -type f -iname "*.xml" | grep -q .; then
    echo "No XML labels found for split: $split"
    echo "Skipping $split"
    return 0
  fi

  if ! find "$split_extract_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | grep -q .; then
    echo "No images found for split: $split"
    echo "Skipping $split"
    return 0
  fi

  echo "Converting $split"
  echo "Input root: $split_extract_dir"
  echo "Output dir: $ROOT_DIR/$split"

  python3 "$CONVERTER" \
    --input-root "$split_extract_dir" \
    --output-dir "$ROOT_DIR/$split"
}

verify_split() {
  local split="$1"

  local img_count
  local label_count

  img_count=$(find "$ROOT_DIR/$split/images" -type f \( \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.png" -o \
    -iname "*.bmp" \
  \) | wc -l)

  label_count=$(find "$ROOT_DIR/$split/labels" -type f -iname "*.txt" | wc -l)

  echo "$split images: $img_count"
  echo "$split labels: $label_count"
}

cleanup_raw() {
  echo "Cleaning everything except .sh, .py, train, val, and test..."

  find "$ROOT_DIR" -mindepth 1 -maxdepth 1 \
    ! -name "train" \
    ! -name "val" \
    ! -name "test" \
    ! -name "*.sh" \
    ! -name "*.py" \
    -exec rm -rf {} +

  echo "Cleanup done."
}

if [[ ! -f "$CONVERTER" ]]; then
  echo "ERROR: Converter script not found:"
  echo "$CONVERTER"
  exit 1
fi

install_requirements
make_output_dirs

echo "Downloading train datasets..."
for link in "${TRAIN_GDRIVE_LINKS[@]}"; do
  download_gdrive_item "$link" "$RAW_DIR/train"
done

echo "Downloading val datasets..."
for link in "${VAL_GDRIVE_LINKS[@]}"; do
  download_gdrive_item "$link" "$RAW_DIR/val"
done

echo "Downloading test datasets..."
for link in "${TEST_GDRIVE_LINKS[@]}"; do
  download_gdrive_item "$link" "$RAW_DIR/test"
done

echo "Extracting train datasets..."
extract_archives "$RAW_DIR/train" "$EXTRACT_DIR/train"

echo "Extracting val datasets..."
extract_archives "$RAW_DIR/val" "$EXTRACT_DIR/val"

echo "Extracting test datasets..."
extract_archives "$RAW_DIR/test" "$EXTRACT_DIR/test"

convert_split "train" "$EXTRACT_DIR/train"
convert_split "val" "$EXTRACT_DIR/val"
convert_split "test" "$EXTRACT_DIR/test"

echo "Verifying final dataset..."
verify_split "train"
verify_split "val"
verify_split "test"

cleanup_raw

echo "Done."