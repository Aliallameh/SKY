import cv2
import numpy as np
import glob
import json
import os


# =========================
# USER SETTINGS
# =========================

# Your printed board:
# Whole squares: 8 x 9
# Inner corners: 7 x 8
CHECKERBOARD_SIZE = (7, 8)  # inner corners: (columns, rows)

# Printed square size:
# 25 mm = 0.025 m
SQUARE_SIZE = 0.025

# Folder containing checkerboard images
IMAGE_FOLDER = "calib_images"

# Output folder
OUTPUT_FOLDER = "Calibration_output"

# Output calibration file
CALIB_FILE = os.path.join(OUTPUT_FOLDER, "calib.json")


def calibrate_camera():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Prepare 3D points of checkerboard corners in real-world coordinates.
    # Since the checkerboard is flat, Z = 0 for all points.
    objp = np.zeros(
        (CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3),
        np.float32
    )

    objp[:, :2] = np.mgrid[
        0:CHECKERBOARD_SIZE[0],
        0:CHECKERBOARD_SIZE[1]
    ].T.reshape(-1, 2)

    objp *= SQUARE_SIZE

    objpoints = []  # 3D points in checkerboard coordinate system
    imgpoints = []  # 2D points in image plane

    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.bmp"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in folder: {IMAGE_FOLDER}")

    print(f"Found {len(image_paths)} images.")

    image_size = None
    valid_images = 0

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    for image_path in image_paths:
        img = cv2.imread(image_path)

        if img is None:
            print(f"[SKIP] Could not read image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_images += 1

            vis = img.copy()
            cv2.drawChessboardCorners(
                vis,
                CHECKERBOARD_SIZE,
                corners_refined,
                found
            )

            filename = os.path.basename(image_path)
            out_path = os.path.join(OUTPUT_FOLDER, f"corners_{filename}")
            cv2.imwrite(out_path, vis)

            print(f"[OK] Corners found: {image_path}")

        else:
            print(f"[FAIL] Corners not found: {image_path}")

    if valid_images < 10:
        raise RuntimeError(
            f"Only {valid_images} valid calibration images found. "
            "Take more images or try CHECKERBOARD_SIZE = (8, 7)."
        )

    print()
    print(f"Valid calibration images: {valid_images}")
    print("Running calibration...")

    reprojection_error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    mean_error = compute_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        K,
        dist
    )

    print()
    print("Calibration complete.")
    print()
    print("OpenCV reprojection error:")
    print(reprojection_error)

    print()
    print("Mean reprojection error:")
    print(mean_error)

    print()
    print("Camera matrix K:")
    print(K)

    print()
    print("Distortion coefficients:")
    print(dist)

    calib_data = {
        "board_description": "8 x 9 full squares, 7 x 8 inner corners",
        "checkerboard_size_inner_corners": {
            "columns": CHECKERBOARD_SIZE[0],
            "rows": CHECKERBOARD_SIZE[1]
        },
        "square_size_m": SQUARE_SIZE,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "opencv_reprojection_error": float(reprojection_error),
        "mean_reprojection_error": float(mean_error),
        "valid_images": valid_images
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

    print()
    print(f"Calibration saved to: {CALIB_FILE}")

    save_undistorted_examples(image_paths, K, dist, image_size)


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            K,
            dist
        )

        error = cv2.norm(
            imgpoints[i],
            projected_points,
            cv2.NORM_L2
        )

        num_points = len(projected_points)
        total_error += error ** 2
        total_points += num_points

    mean_error = np.sqrt(total_error / total_points)
    return mean_error


def save_undistorted_examples(image_paths, K, dist, image_size):
    print()
    print("Saving undistorted sample images...")

    w, h = image_size

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (w, h),
        alpha=1,
        newImgSize=(w, h)
    )

    for i, image_path in enumerate(image_paths[:5]):
        img = cv2.imread(image_path)

        if img is None:
            continue

        undistorted = cv2.undistort(
            img,
            K,
            dist,
            None,
            new_K
        )

        filename = os.path.basename(image_path)

        original_out = os.path.join(
            OUTPUT_FOLDER,
            f"original_sample_{i}_{filename}"
        )

        undistorted_out = os.path.join(
            OUTPUT_FOLDER,
            f"undistorted_sample_{i}_{filename}"
        )

        cv2.imwrite(original_out, img)
        cv2.imwrite(undistorted_out, undistorted)

    print("Undistorted sample images saved.")


if __name__ == "__main__":
    calibrate_camera()