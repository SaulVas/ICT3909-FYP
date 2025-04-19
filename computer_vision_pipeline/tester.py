# pylint: disable=no-member
import os
import cv2
import numpy as np
from pathlib import Path


def process_images():
    # Get the current directory
    current_dir = Path(__file__).parent
    outputs_dir = current_dir.parent / "outputs"

    # Walk through all subdirectories
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file.endswith("_overlay.png"):
                # Construct full file path
                file_path = os.path.join(root, file)

                # Read the image
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Could not read image: {file_path}")
                    continue

                # Get image dimensions
                height, width = img.shape[:2]

                # Draw a horizontal line in the middle
                # Using red color (BGR format) and thickness of 2 pixels
                cv2.line(img, (0, height // 2), (width, height // 2), (0, 0, 255), 2)

                # Save the modified image, overwriting the original
                cv2.imwrite(file_path, img)
                print(f"Processed: {file_path}")


if __name__ == "__main__":
    process_images()
