#!/usr/bin/env python3
"""Description: Exercise 1 - Download image, generate max 32x32 matrix table,

and display rotated high-res image.
"""

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/e/e4/Lemon.jpg"
OUTPUT_DIR = Path("data")
RAW_IMAGE_PATH = OUTPUT_DIR / "lemon_raw.jpg"
PROCESSED_IMAGE_PATH = OUTPUT_DIR / "lemon_rotated.jpg"


def download_image(url: str, output_path: Path) -> Path:
    """Download a remote image from a URL to a local path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"Raw image already exists locally at: {output_path}")
        return output_path

    logger.info(f"Downloading image from: {url}")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GKiPO-Python-App/1.0 (Educational)"},
        )
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as out_file:
            out_file.write(resp.read())

        logger.info(f"Successfully saved raw image to: {output_path}")
        return output_path
    except (urllib.error.URLError, OSError) as e:
        logger.exception(f"Failed to download image due to network or IO error: {e}")
        raise


def process_image(
    input_path: Path,
    output_path: Path | None = None,
    max_dim: int | None = None,
) -> np.ndarray:
    """Process image: optional scaling, greyscale, and 90° rotation."""
    logger.info(f"Processing image (max_dim={max_dim}): {input_path}")

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read the image at {input_path}")

    orig_h, orig_w = img.shape[:2]

    if max_dim is not None:
        scale = min(max_dim / orig_w, max_dim / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
    else:
        new_w = max(1, int(orig_w * 0.5))
        new_h = max(1, int(orig_h * 0.5))

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)

    rot_height, rot_width = rotated_img.shape
    logger.info(f"Processed dimensions (Width x Height): {rot_width}x{rot_height}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), rotated_img)
        if not success:
            raise OSError(f"Failed to write processed image to {output_path}")
        logger.info(f"Successfully saved processed image to: {output_path}")

    return rotated_img


def print_matrix_table(matrix: np.ndarray) -> None:
    """Print full low-res pixel matrix as a formatted table of numbers."""
    rows, cols = matrix.shape
    logger.info(f"Displaying full {rows}x{cols} low-res pixel matrix:")

    header = " | ".join([f"{i:3d}" for i in range(cols)])
    separator = "----+" + "-" * (len(header) + 1)

    print(f"\nCol  | {header}")
    print(separator)
    for idx, row in enumerate(matrix):
        row_str = " | ".join([f"{val:3d}" for val in row])
        print(f"Row {idx:2d} | {row_str}")
    print()


def display_image(image_path: Path) -> None:
    """Display rotated greyscale image with canvas matching source size."""
    logger.info(f"Opening high-res rotated image for display: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read the image at {image_path}")

    height, width = img.shape
    dpi = plt.rcParams["figure.dpi"]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    plt.show()


def main() -> None:
    """Core execution flow for Exercise 1."""
    try:
        raw_path = download_image(DEFAULT_IMAGE_URL, RAW_IMAGE_PATH)
        low_res_matrix = process_image(raw_path, output_path=None, max_dim=32)
        print_matrix_table(low_res_matrix)
        process_image(raw_path, output_path=PROCESSED_IMAGE_PATH, max_dim=None)
        display_image(PROCESSED_IMAGE_PATH)
    except (
        urllib.error.URLError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as e:
        logger.error(f"Exercise 1 execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
