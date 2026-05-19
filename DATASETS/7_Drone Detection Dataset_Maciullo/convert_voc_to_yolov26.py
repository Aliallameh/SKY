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
        description="Convert Drone Pascal VOC XML dataset to YOLO format."
    )

    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Path to root folder containing DroneTrainDataset and DroneTestDataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output YOLO dataset directory"
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Ratio of training data to use for validation. Default: 0.2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid split"
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    train_images_dir = input_root / "DroneTrainDataset" / "Drone_TrainSet"
    train_xmls_dir = input_root / "DroneTrainDataset" / "Drone_TrainSet_XMLs"

    test_images_dir = input_root / "DroneTestDataset" / "Drone_TestSet"
    test_xmls_dir = input_root / "DroneTestDataset" / "Drone_TestSet_XMLs"

    if not train_images_dir.exists():
        raise FileNotFoundError(f"Training images folder not found: {train_images_dir}")

    if not train_xmls_dir.exists():
        raise FileNotFoundError(f"Training XML folder not found: {train_xmls_dir}")

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images folder not found: {test_images_dir}")

    if not test_xmls_dir.exists():
        raise FileNotFoundError(f"Test XML folder not found: {test_xmls_dir}")

    make_dirs(output_dir)

    train_samples_all = collect_samples(train_images_dir, train_xmls_dir)
    test_samples = collect_samples(test_images_dir, test_xmls_dir)

    random.seed(args.seed)
    random.shuffle(train_samples_all)

    valid_count = int(len(train_samples_all) * args.valid_ratio)

    valid_samples = train_samples_all[:valid_count]
    train_samples = train_samples_all[valid_count:]

    print(f"Total original train samples: {len(train_samples_all)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Test samples: {len(test_samples)}")

    copy_and_convert_samples(train_samples, output_dir, "train")
    copy_and_convert_samples(valid_samples, output_dir, "valid")
    copy_and_convert_samples(test_samples, output_dir, "test")

    write_data_yaml(output_dir)

    print("\nConversion completed successfully.")
    print(f"YOLO dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()