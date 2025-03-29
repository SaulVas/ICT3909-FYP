# pylint: disable=no-member

import os
from pathlib import Path
from functools import partial
import cv2
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class SplineDetector:
    def __init__(self, input_images: list[str]):
        self.image_paths = input_images
        # Create output directory if it doesn't exist
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

        for image_path in self.image_paths:
            self._extract_spline_data(image_path)

    def _extract_spline_data(self, image_path: str) -> np.ndarray:
        base_name = Path(image_path).stem
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(
                f"Failed to load image at '{image_path}'. "
                "Please ensure the file exists and is a valid image format."
            )

        red_pixels = self._extract_red_pixels(image)
        gray_scale = cv2.cvtColor(red_pixels, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray_scale, (25, 25), 0)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=10)

        edges, contours = self._extract_edges(dilated)

        mask = self._create_height_mask(edges, contours)

        # sort contours by length
        get_length = partial(self._get_masked_contour_length, edges, mask=mask)
        sorted_contours = sorted(contours, key=get_length, reverse=True)

        colored_edges, spline_points = self._draw_splines(
            edges, sorted_contours, mask, [RED, GREEN, BLUE]
        )

        # Draw lines between points(only for visual purposes)
        for color_idx in range(3):
            points = spline_points[color_idx]
            if len(points) > 0:
                sorted_indices = np.argsort(points[:, 0])
                y_coords = points[sorted_indices, 1]
                x_coords = points[sorted_indices, 0]

                for j in range(len(x_coords) - 1):
                    pt1 = (x_coords[j], y_coords[j])
                    pt2 = (x_coords[j + 1], y_coords[j + 1])
                    cv2.line(colored_edges, pt1, pt2, RED, 2)

        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 1.0, colored_edges, 1.0, 0)
        self._save_image(
            base_name,
            image=image,
            overlay=overlay,
        )

        return spline_points

    def _extract_red_pixels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Convert directly to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Expanded HSV ranges to include all values
        ranges = [
            # Red at start of hue circle (0-15)
            ([0, 5, 20], [15, 255, 255]),  # Lowered S minimum to catch desaturated reds
            # Purple/magenta through red range (135-180)
            ([135, 5, 20], [180, 255, 255]),  # Expanded range to catch all distant reds
        ]

        # Create and combine all masks
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply mask to get only red pixels
        red_pixels = cv2.bitwise_and(image, image, mask=combined_mask)

        return red_pixels

    def _extract_edges(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, threshold1=50, threshold2=150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No contours found in the edge image")

        return edges, contours

    def _create_height_mask(
        self, edges: np.ndarray, contours: list[np.ndarray]
    ) -> np.ndarray:
        mask = np.zeros_like(edges)

        largest_contour = max(contours, key=cv2.contourArea)
        longest_contour_mask = np.zeros_like(edges)
        cv2.drawContours(longest_contour_mask, [largest_contour], -1, 255, -1)

        longest_y_coords = np.where(longest_contour_mask > 0)[0]
        if len(longest_y_coords) > 0:
            for x in range(edges.shape[1]):
                column_mask = longest_contour_mask[:, x]
                y_coords = np.where(column_mask > 0)[0]
                if len(y_coords) > 0:
                    mask[: y_coords[-1], x] = 255

        return mask

    def _get_masked_contour_length(
        self, edges: np.ndarray, contour: np.ndarray, mask: np.ndarray
    ) -> int:
        temp_mask = np.zeros_like(edges)
        cv2.drawContours(temp_mask, [contour], -1, 255, 2)
        masked_contour = cv2.bitwise_and(temp_mask, mask)

        return np.count_nonzero(masked_contour)

    def _average_y_coordinates(self, masked_points: np.ndarray) -> np.ndarray:
        y_coords = masked_points[0]
        x_coords = masked_points[1]

        # Create a dictionary to store all y values for each x coordinate
        x_to_y_map = {}
        for x, y in zip(x_coords, y_coords):
            if x not in x_to_y_map:
                x_to_y_map[x] = []
            x_to_y_map[x].append(y)

        # Calculate average y for each x
        averaged_points = np.zeros((2, len(x_to_y_map)), dtype=np.int32)
        for idx, (x, y_list) in enumerate(x_to_y_map.items()):
            averaged_points[1, idx] = x  # x coordinate
            averaged_points[0, idx] = int(np.mean(y_list))  # average y coordinate

        return tuple(averaged_points)

    def _draw_splines(
        self,
        edges: np.ndarray,
        sorted_contours: list[np.ndarray],
        mask: np.ndarray,
        colors: list[tuple[int, int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draws splines and returns both the colored edge image and spline coordinates.

        Args:
            edges: Base image to determine dimensions
            sorted_contours: List of contours sorted by length
            mask: Height mask for filtering contours
            colors: List of BGR colors for each spline

        Returns:
            tuple: (colored_edges image, spline_points)
                - colored_edges: np.ndarray of the visualization
                - spline_points: np.ndarray of shape (3,) containing arrays of points for
                                 each spline
                    Each interior array has shape (N, 2) where N is the number of points
                    Format: [[x1,y1], [x2,y2], ...] for each spline
        """
        colored_edges = np.zeros_like(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        spline_points = np.empty(3, dtype=object)  # Array to hold the 3 splines' points

        # Process each contour (up to 3)
        for i, contour in enumerate(sorted_contours[:3]):
            contour_mask = np.zeros_like(edges)
            cv2.drawContours(contour_mask, [contour], -1, 255, 2)
            masked_contour = cv2.bitwise_and(contour_mask, mask)

            # Find the masked contour points and average them
            masked_points = np.where(masked_contour > 0)
            averaged_points = self._average_y_coordinates(masked_points)

            # Convert to (N,2) array format and store
            points_array = np.column_stack(
                (averaged_points[1], averaged_points[0])
            )  # [[x1,y1], [x2,y2], ...]
            spline_points[i] = points_array

            # Draw the spline (for visualization)
            colored_edges[averaged_points] = colors[i]

            # Draw the lowest visible point
            if points_array.size > 0:
                max_y_idx = np.argmax(
                    points_array[:, 1]
                )  # Find point with largest y value
                lowest_visible_point = tuple(points_array[max_y_idx])
                cv2.circle(colored_edges, lowest_visible_point, 5, colors[i], -1)

        return colored_edges, spline_points

    def _save_image(
        self,
        base_name: str,
        image: np.ndarray = None,
        gray_scale: np.ndarray = None,
        blurred: np.ndarray = None,
        dilated: np.ndarray = None,
        binary: np.ndarray = None,
        colored_edges: np.ndarray = None,
        overlay: np.ndarray = None,
    ):
        if image is not None:
            cv2.imwrite(str(self.output_dir / f"{base_name}_original.png"), image)

        if gray_scale is not None:
            cv2.imwrite(
                str(self.output_dir / f"{base_name}_gray_red_pixels.png"),
                gray_scale,
            )

        if blurred is not None:
            cv2.imwrite(str(self.output_dir / f"{base_name}_blurred.png"), blurred)

        if dilated is not None:
            cv2.imwrite(str(self.output_dir / f"{base_name}_dilated.png"), dilated)

        if binary is not None:
            cv2.imwrite(str(self.output_dir / f"{base_name}_binary.png"), binary)

        if colored_edges is not None:
            cv2.imwrite(
                str(self.output_dir / f"{base_name}_splines.png"), colored_edges
            )

        if overlay is not None:
            cv2.imwrite(str(self.output_dir / f"{base_name}_overlay.png"), overlay)


if __name__ == "__main__":
    # Define the dataset path
    DATASET_PATH = "data/on_water_dataset"

    # Check if the directory exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found")

    # Find all images recursively
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(str(p) for p in Path(DATASET_PATH).rglob(ext))

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in '{DATASET_PATH}' or its subdirectories"
        )

    print(f"Found {len(image_files)} images")

    # Initialize the SplineDetector with the image files
    detector = SplineDetector(image_files)
