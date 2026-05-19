import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_NAMES = ["drone"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def voc_bbox_to_yolo(size, bbox):
    """
    Convert Pascal VOC bbox to YOLO format.

    VOC:
        xmin, ymin, xmax, ymax

    YOLO:
        class_id x_center y_center width height

    All YOLO values are normalized between 0 and 1.
    """
    img_w, img_h = size
    xmin, ymin, xmax, ymax = bbox

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h

    return x_center, y_center, box_w, box_h


def parse_voc_xml(xml_path):
    """
    Parse one Pascal VOC XML file and return:
        image filename
        image width
        image height
        list of YOLO label lines
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")

    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[class_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        # Clamp bbox to image boundaries
        xmin = max(0, min(xmin, img_w))
        ymin = max(0, min(ymin, img_h))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))

        # Skip invalid boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, box_w, box_h = voc_bbox_to_yolo(
            size=(img_w, img_h),
            bbox=(xmin, ymin, xmax, ymax)
        )

        yolo_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_w:.6f} "
            f"{box_h:.6f}"
        )

        yolo_lines.append(yolo_line)

    return filename, yolo_lines


def make_dirs(output_dir):
    """
    Create YOLO dataset folders.
    """
    for split in ["train", "valid", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def collect_samples(images_dir, xmls_dir):
    """
    Match images with XML files based on stem name.

    Example:
        2.jpg  <-->  2.xml
        VS_N10.jpg  <-->  VS_N10.xml
    """
    images_dir = Path(images_dir)
    xmls_dir = Path(xmls_dir)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    samples = []

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        xml_path = xmls_dir / f"{img_path.stem}.xml"

        if xml_path.exists():
            samples.append((img_path, xml_path))
        else:
            print(f"Warning: XML not found for image: {img_path.name}")

    return samples


def copy_and_convert_samples(samples, output_dir, split_name):
    """
    Copy images and write YOLO .txt labels.
    """
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"

    for img_path, xml_path in samples:
        try:
            filename_from_xml, yolo_lines = parse_voc_xml(xml_path)

            # Prefer actual image file name, because it is guaranteed to exist
            output_image_name = img_path.name
            output_label_name = f"{img_path.stem}.txt"

            shutil.copy2(img_path, images_out / output_image_name)

            label_path = labels_out / output_label_name
            with open(label_path, "w", encoding="utf-8") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines))
                    f.write("\n")

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")


def write_data_yaml(output_dir):
    """
    Write data.yaml for YOLO training.
    """
    yaml_content = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['drone']

roboflow:
  workspace: dronews
  project: lrddv2-fhowg-cegrh
  version: 4
  license: Public Domain
  url: https://universe.roboflow.com/dronews/lrddv2-fhowg-cegrh/dataset/4
"""

    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML dataset to YOLO format."
    )

    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--xml-dir", type=str, required=True)
    parser.add_argument("--output-images-dir", type=str, required=True)
    parser.add_argument("--output-labels-dir", type=str, required=True)
    parser.add_argument("--classes", nargs="+", default=["drone"])

    args = parser.parse_args()

    global CLASS_NAMES, CLASS_TO_ID
    CLASS_NAMES = args.classes
    CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    images_dir = Path(args.images_dir)
    xmls_dir = Path(args.xml_dir)
    output_images_dir = Path(args.output_images_dir)
    output_labels_dir = Path(args.output_labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not xmls_dir.exists():
        raise FileNotFoundError(f"XML folder not found: {xmls_dir}")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(images_dir, xmls_dir)

    print(f"Images dir: {images_dir}")
    print(f"XML dir: {xmls_dir}")
    print(f"Matched samples: {len(samples)}")

    for img_path, xml_path in samples:
        try:
            _, yolo_lines = parse_voc_xml(xml_path)

            output_image_name = img_path.name
            output_label_name = f"{img_path.stem}.txt"

            shutil.copy2(img_path, output_images_dir / output_image_name)

            label_path = output_labels_dir / output_label_name
            with open(label_path, "w", encoding="utf-8") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines))
                    f.write("\n")

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")

    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()