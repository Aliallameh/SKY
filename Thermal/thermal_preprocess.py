'''
Linux/WSL:

requirements:
python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

echo "opencv-python" > requirements.txt

pip install -r requirements.txt

running:

python split_thermal_video.py \
  --input your_video.mp4 \



Windows:

requirements:

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

"opencv-python" | Out-File -Encoding ascii requirements.txt

pip install -r requirements.txt

running:

python split_thermal_video.py --input thermal_video.mp4 --output-dir split_results

'''

import cv2
import os
import argparse


def split_video(input_path, output_dir="output_split", split_y=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   



    print(f"Input video: {width}x{height}, FPS={fps}")

    # If you do not provide split_y, assume bottom half starts around 2/3 height.
    # For your screenshot, the split seems around y = 700 in a 1024 px image.

    split_y = height // 2

    print(f"Using split_y = {split_y}")

    top_h = split_y
    bottom_h = height - split_y

    basename = os.path.splitext(os.path.basename(input_path))[0]

    top_output = os.path.join(output_dir, f"{basename}_top.mp4")
    bottom_output = os.path.join(output_dir, f"{basename}_bottom.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    top_writer = cv2.VideoWriter(top_output, fourcc, fps, (width, top_h))
    bottom_writer = cv2.VideoWriter(bottom_output, fourcc, fps, (width, bottom_h))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        top_frame = frame[:split_y, :]
        bottom_frame = frame[split_y:, :]

        top_writer.write(top_frame)
        bottom_writer.write(bottom_frame)

        frame_count += 1

    cap.release()
    top_writer.release()
    bottom_writer.release()

    print(f"Done. Processed {frame_count} frames.")
    print(f"Top video saved to: {top_output}")
    print(f"Bottom video saved to: {bottom_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input mp4 video")
    parser.add_argument("--output-dir", default="output_split", help="Folder to save output videos")
    parser.add_argument("--split-y", type=int, default=None, help="Y pixel where bottom video starts")

    args = parser.parse_args()

    split_video(
        input_path=args.input,
        output_dir=args.output_dir,
        split_y=args.split_y
    )