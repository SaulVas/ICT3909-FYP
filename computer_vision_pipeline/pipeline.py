# pylint: disable=no-member

import os
from pathlib import Path
import cv2
import numpy as np

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

COLOURS = [YELLOW, GREEN, BLUE]


class SplineDetector:
    def __init__(self, input_images: list[str]):
        self.image_paths = input_images
        # Create base output directory if it doesn't exist
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
        eroded = cv2.erode(dilated, kernel, iterations=5)

        edges, contours_all, _ = self._extract_edges(eroded)

        # Find the prioritized contours
        prioritized_contours = self._find_contours_by_priority(contours_all)

        # Calculate spline points from the selected contours
        spline_points = self._calculate_spline_points(prioritized_contours)

        # Create visualization of the splines
        colored_edges = self._create_spline_visualization(edges, spline_points, COLOURS)

        # Draw lines between points(only for visual purposes)
        for color_idx, contour in enumerate(prioritized_contours):
            if contour is None:
                continue
            points = spline_points[color_idx]
            if points is not None and len(points) > 1:
                # Sort points by x-coordinate for line drawing
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]

                for j in range(len(sorted_points) - 1):
                    pt1 = tuple(sorted_points[j].astype(int))
                    pt2 = tuple(sorted_points[j + 1].astype(int))
                    cv2.line(colored_edges, pt1, pt2, COLOURS[color_idx], 2)

        overlay = image.copy()
        overlay = cv2.addWeighted(overlay, 1.0, colored_edges, 1.0, 0)
        self._save_image(
            base_name,
            image_path,
            edges=edges,
            image=image,
            colored_edges=colored_edges,
            overlay=overlay,
        )

        # Ensure spline_points has the correct structure even with None contours
        final_spline_points = np.array(
            [pts if pts is not None else np.array([]) for pts in spline_points],
            dtype=object,
        )
        print(f"Processed {image_path}")
        return final_spline_points

    def _extract_red_pixels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Convert directly to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for the upper 2/3 of the image
        height = image.shape[0]
        upper_height = int(height * 2 / 3)
        upper_mask = np.zeros_like(hsv_image[:, :, 0])
        upper_mask[:upper_height, :] = 255

        # Expanded HSV ranges to include all values
        ranges = [
            # Red at start of hue circle (0-15)
            ([0, 3, 20], [15, 255, 255]),  # Lowered S minimum to catch desaturated reds
            # Purple/magenta through red range (135-180)
            ([145, 3, 20], [180, 255, 255]),
        ]

        # Create and combine all masks
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Combine with upper mask
        combined_mask = cv2.bitwise_and(combined_mask, upper_mask)

        # Apply mask to get only red pixels
        red_pixels = cv2.bitwise_and(image, image, mask=combined_mask)

        return red_pixels

    def _extract_edges(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(binary, threshold1=50, threshold2=150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # Return empty list instead of raising error if no contours found
            print("Warning: No contours found in the edge image")
            return edges, [], np.zeros_like(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

        # Create a visualization of the contours
        contour_vis = np.zeros_like(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)

        return edges, contours, contour_vis

    def _get_horizontal_extent(self, contour: np.ndarray) -> int:
        """Calculates the horizontal extent (width) of a contour."""
        if contour is None or len(contour) == 0:
            return 0
        min_x = np.min(contour[:, :, 0])
        max_x = np.max(contour[:, :, 0])
        return max_x - min_x

    def _get_highest_y_coordinate(self, contour: np.ndarray) -> int | None:
        """Calculates the highest y-coordinate of a contour.
        Returns None if the contour is empty or None.

        (highest y-coordinate, lowest y-coordinate as (0, 0) is top left)
        """
        if contour is None or len(contour) == 0:
            return None
        highest_cord = np.min(contour[:, :, 1])
        return highest_cord

    def _find_contours_by_priority(self, contours: list[np.ndarray]) -> tuple[
        list[np.ndarray | None],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],  # Added return type for candidates_2 list
    ]:
        """
        Finds three contours based on horizontal extent and vertical positioning.

        1. Longest horizontal extent.
        2. Longest horizontal extent among contours entirely above the first.
        3. Longest horizontal extent among contours entirely above the second.

        Args:
            contours: List of contours found in the image.

        Returns:
            A tuple containing:
            - selected_contours: List containing up to three selected contours [c1, c2, c3],
                                 with None placeholders if fewer than three are found.
        """
        empty_return = [None, None, None]
        if not contours:
            return empty_return

        # Calculate extents and store with contours
        contours_with_extents = []
        for c in contours:
            extent = self._get_horizontal_extent(c)
            if extent > 0:  # Ignore points or vertical lines if necessary
                contours_with_extents.append((extent, c))

        if not contours_with_extents:
            return empty_return

        # Sort by extent descending
        contours_with_extents.sort(key=lambda item: item[0], reverse=True)

        selected_contours = [None, None, None]

        available_contours_data = list(contours_with_extents)  # Use list copy

        # --- Select Contour 1 ---
        if not available_contours_data:
            return empty_return  # Should not happen here, but defensive check

        selected_contours[0] = available_contours_data[0][1]
        available_contours_data.pop(0)

        top_y1 = self._get_highest_y_coordinate(selected_contours[0])
        if top_y1 is None:  # Should not happen if extent > 0 but handle defensively
            # Only C1 selected
            return selected_contours

        # --- Select Contour 2 ---
        candidates_2_data = []  # Store tuples (extent, contour)

        for extent, c in available_contours_data:
            top_yc = self._get_highest_y_coordinate(c)
            # Ensure contour has points before checking bounds
            if top_yc is not None and top_yc < top_y1:
                candidates_2_data.append((extent, c))

        if candidates_2_data:  # Check if the data list (with extents) is populated
            candidates_2_data.sort(key=lambda item: item[0], reverse=True)
            selected_contours[1] = candidates_2_data[0][1]
            candidates_2_data.pop(0)

            top_y2 = self._get_highest_y_coordinate(selected_contours[1])
            if top_y2 is None:
                # Only C1, C2 selected, C3 remaining list is same as C2's
                return selected_contours

            # --- Select Contour 3 ---
            candidates_3 = []
            for extent, c in candidates_2_data:
                top_yc = self._get_highest_y_coordinate(c)
                if top_yc is not None and top_yc < top_y2:
                    candidates_3.append((extent, c))

            if candidates_3:
                candidates_3.sort(key=lambda item: item[0], reverse=True)
                selected_contours[2] = candidates_3[0][1]

        return selected_contours

    def _calculate_spline_points(
        self, selected_contours: list[np.ndarray | None]
    ) -> np.ndarray:
        """Calculates the averaged spline points for selected contours."""
        spline_points = np.empty(3, dtype=object)
        for i, contour in enumerate(selected_contours):
            if contour is None:
                spline_points[i] = None
                continue
            averaged_points_coords = self._average_y_coordinates(contour)
            if averaged_points_coords is None:
                spline_points[i] = None
                continue
            points_array = np.column_stack(
                (averaged_points_coords[1], averaged_points_coords[0])
            )
            spline_points[i] = points_array
        return spline_points

    def _create_spline_visualization(
        self,
        edges: np.ndarray,
        spline_points: np.ndarray,  # Takes the calculated points
        colors: list[tuple[int, int, int]],
    ) -> np.ndarray:
        """Creates an image visualizing the calculated spline points."""
        colored_edges = np.zeros_like(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        for i, points_array in enumerate(spline_points):
            if points_array is None or points_array.size == 0:
                continue

            # Average points are already calculated, need coords for drawing
            # points_array is (N, 2) -> [[x1,y1], [x2,y2], ...]
            avg_y = points_array[:, 1].astype(np.int32)
            avg_x = points_array[:, 0].astype(np.int32)
            averaged_points_coords = (avg_y, avg_x)  # Recreate tuple for indexing

            # Draw the spline points
            colored_edges[averaged_points_coords] = colors[i]

            # Draw the lowest visible point of the spline
            max_y_idx = np.argmax(points_array[:, 1])
            lowest_visible_point = tuple(points_array[max_y_idx].astype(int))
            cv2.circle(colored_edges, lowest_visible_point, 5, colors[i], -1)

        return colored_edges

    def _average_y_coordinates(self, contour: np.ndarray) -> np.ndarray | None:
        """Averages Y coordinates for each X coordinate in a contour."""
        if contour is None or len(contour) == 0:
            return None

        # Extract points: contour shape is (N, 1, 2) -> get (N, 2)
        points = contour.squeeze(axis=1)
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # Create a dictionary to store all y values for each x coordinate
        x_to_y_map = {}
        for x, y in zip(x_coords, y_coords):
            if x not in x_to_y_map:
                x_to_y_map[x] = []
            x_to_y_map[x].append(y)

        # Calculate average y for each x
        if not x_to_y_map:
            return None  # No points to average

        num_averaged = len(x_to_y_map)
        avg_y = np.zeros(num_averaged, dtype=np.int32)
        avg_x = np.zeros(num_averaged, dtype=np.int32)

        for idx, (x, y_list) in enumerate(x_to_y_map.items()):
            avg_x[idx] = x
            avg_y[idx] = int(np.mean(y_list))

        # Return coordinates in (y_coords, x_coords) tuple format for drawing
        return (avg_y, avg_x)

    def _get_masked_contour_length(
        self, edges: np.ndarray, contour: np.ndarray, mask: np.ndarray
    ) -> int:
        # This function might be obsolete now or need adjustment if still used elsewhere
        temp_mask = np.zeros_like(edges)
        cv2.drawContours(temp_mask, [contour], -1, 255, 2)
        masked_contour = cv2.bitwise_and(temp_mask, mask)

        return np.count_nonzero(masked_contour)

    def _save_image(
        self,
        base_name: str,
        image_path: str,
        image: np.ndarray = None,
        gray_scale: np.ndarray = None,
        blurred: np.ndarray = None,
        dilated: np.ndarray = None,
        eroded: np.ndarray = None,
        edges: np.ndarray = None,
        contour_vis: np.ndarray = None,
        colored_edges: np.ndarray = None,
        overlay: np.ndarray = None,
    ):
        # Create subdirectory structure based on input path
        input_path = Path(image_path)
        # Get the parent directory name (e.g., '107' from 'data/on_water_dataset/107/IMG_3048.jpg')
        subdir_name = input_path.parent.name
        # Create the output subdirectory
        output_subdir = self.output_dir / subdir_name
        output_subdir.mkdir(exist_ok=True)

        if image is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_original.png"), image)

        if gray_scale is not None:
            cv2.imwrite(
                str(output_subdir / f"{base_name}_gray_red_pixels.png"),
                gray_scale,
            )

        if blurred is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_blurred.png"), blurred)

        if dilated is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_dilated.png"), dilated)

        if eroded is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_eroded.png"), eroded)

        if edges is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_edges.png"), edges)

        if contour_vis is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_contours.png"), contour_vis)

        if colored_edges is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_splines.png"), colored_edges)

        if overlay is not None:
            cv2.imwrite(str(output_subdir / f"{base_name}_overlay.png"), overlay)


if __name__ == "__main__":
    # Define the dataset path
    DATASET_PATH = "data/debug_dataset"

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
