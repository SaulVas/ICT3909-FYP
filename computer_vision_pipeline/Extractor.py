import math
from collections import defaultdict
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import os


class Extractor:
    def __init__(self):
        pass

    def __call__(
        self,
        image_splines: np.ndarray,
        image_path: str,
        num_new_points: int = 30,
        debugging: bool = False,
    ) -> np.ndarray:
        if len(image_splines) != 3:
            print(f"Image: {image_path} has {len(image_splines)} splines.")
            return None

        tcd_data = defaultdict(tuple)
        primary_twist = 0.0
        for index, spline in enumerate(image_splines):
            if len(spline) < num_new_points:
                print(f"Spline {index} has less than {num_new_points} points.")
                tcd_data[index] = ((0.0, 0.0, 0.0), 0.0, [])
                continue

            reduced_spline = self._interpolate_spline_points(spline, num_new_points)

            if debugging:
                self._debug_interpolated_spline(
                    reduced_spline, spline, image_path, index, num_new_points
                )

            if reduced_spline is None:
                print(f"Skipping spline {index} due to interpolation failure.")
                continue

            tcd_result, straight_line_length = self._extract_tcd(
                reduced_spline, num_new_points
            )

            if tcd_result is not None and straight_line_length > 0:
                if index == 0:
                    primary_twist = tcd_result[0]

                processed_tcd = (
                    np.abs(tcd_result[0] - primary_twist),
                    tcd_result[1],
                    tcd_result[2],
                )
                tcd_data[index] = (
                    processed_tcd,
                    straight_line_length,
                    reduced_spline.tolist(),
                )

        return dict(tcd_data)

    def _interpolate_spline_points(
        self, spline: np.ndarray, num_new_points: int
    ) -> np.ndarray | None:
        if not isinstance(spline, np.ndarray) or len(spline) < num_new_points:
            print(
                f"Invalid spline input for interpolation: Must be a Nx2 numpy array with N >= 2. Got shape: {spline.shape if isinstance(spline, np.ndarray) else type(spline)}"
            )
            return np.ndarray([])

        t = np.linspace(0, 1, spline.shape[0])
        t_new = np.linspace(0, 1, num_new_points)

        x_interp = interp.CubicSpline(t, spline[:, 0], bc_type="natural")
        y_interp = interp.CubicSpline(t, spline[:, 1], bc_type="natural")

        new_x = x_interp(t_new)
        new_y = y_interp(t_new)

        interpolated_points = np.column_stack((new_x, new_y))
        interpolated_points[0] = spline[0]
        interpolated_points[-1] = spline[-1]

        return interpolated_points

    def _extract_tcd(self, spline: np.ndarray, num_new_points: int) -> tuple:
        if spline is None or len(spline) < num_new_points:
            return (0.0, 0.0, 0.0), 0.0

        start_point = spline[0]
        end_point = spline[-1]

        straight_line = np.subtract(end_point, start_point)
        x_axis = np.array([1, 0])

        # -- TWIST -- #
        twist_radians = np.arccos(
            np.divide(
                np.dot(straight_line, x_axis),
                np.multiply(np.linalg.norm(straight_line), np.linalg.norm(x_axis)),
            )
        )

        twist = 180 - np.degrees(twist_radians)

        # -- CAMBER -- #
        camber_distance = 0
        camber_point = []
        for index, point in enumerate(spline):
            if index in [0, len(spline) - 1]:
                continue

            distance = np.divide(
                np.linalg.norm(
                    np.cross(
                        np.subtract(point, start_point), np.subtract(point, end_point)
                    )
                ),
                np.linalg.norm(straight_line),
            )

            if distance > camber_distance:
                camber_distance = distance
                camber_point = point

        camber = np.divide(camber_distance, np.linalg.norm(straight_line)) * 100

        # -- DRAUGHT -- #
        camber_point_hypotenuse = np.linalg.norm(np.subtract(camber_point, start_point))

        draught_distance = math.sqrt(camber_point_hypotenuse**2 - camber_distance**2)
        draught = np.divide(draught_distance, np.linalg.norm(straight_line)) * 100

        return (twist, camber, draught), np.linalg.norm(straight_line)

    def _debug_interpolated_spline(
        self,
        reduced_spline: np.ndarray,
        original_spline: np.ndarray,
        image_path: str,
        index: int,
        num_new_points: int,
    ):
        if (
            len(reduced_spline) < num_new_points
            or len(original_spline) < num_new_points
        ):
            return

        fig, ax = plt.subplots()
        ax.plot(
            original_spline[:, 0],
            original_spline[:, 1],
            "bo-",
            label="Original Spline",
            markersize=6,
        )
        ax.plot(
            reduced_spline[:, 0],
            reduced_spline[:, 1],
            "r.--",
            label="Interpolated Spline",
            markersize=4,
        )

        # Extract base filename and create directory
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = os.path.join("debugging_plots", base_filename)
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, f"spline_{index}_debug.png")

        ax.set_title(f"Spline {index} - {base_filename}")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        plt.savefig(save_path)
        plt.close(fig)
