#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Windows Git Bash script
#
# Run from inside 7_DroneDetectionDataset using Git Bash:
#
#   bash download_and_convert_windows_gitbash.sh
#
# Do NOT run this directly in PowerShell.
# ============================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

PYTHON_CMD="python"

install_requirements() {
  "$PYTHON_CMD" -m pip install --upgrade pip
  "$PYTHON_CMD" -m pip install gdown tqdm
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
  gdown --fuzzy "$link" -O "$output_dir/"
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
        echo "ERROR: .rar extraction is not supported by default in Git Bash."
        echo "Please extract the .rar manually or install unrar/7zip and modify this block."
        exit 1
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

  local image_dir
  local xml_dir

  image_dir="$(find_image_dir "$split_extract_dir" || true)"
  xml_dir="$(find_xml_dir "$split_extract_dir" || true)"

  if [[ -z "${image_dir:-}" || -z "${xml_dir:-}" ]]; then
    echo "No images/XML labels found for split: $split"
    echo "Skipping $split"
    return 0
  fi

  echo "Converting $split"
  echo "Images: $image_dir"
  echo "XMLs:   $xml_dir"

  "$PYTHON_CMD" "$CONVERTER" \
    --images-dir "$image_dir" \
    --xml-dir "$xml_dir" \
    --output-images-dir "$ROOT_DIR/$split/images" \
    --output-labels-dir "$ROOT_DIR/$split/labels" \
    --classes "${CLASSES[@]}"
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
  echo "Deleting raw downloads/extracted folders..."
  rm -rf "$RAW_DIR" "$EXTRACT_DIR"
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