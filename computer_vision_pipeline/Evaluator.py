import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors


class Evaluator:
    def __init__(self, spline_data, ground_truth):
        self.spline_data = spline_data
        self.ground_truth = ground_truth
        # let l be the length from the first to last point of the top spline
        self.l_vals = defaultdict(list)
        self.reduced_spline_data = {}
        self.images_used = {}
        self.subdir_means = {}
        self.global_mean = 0.0
        self.global_std = 0.0
        self.subdir_errors = {}
        self.threshold = 0.0

    def __call__(self, csv_path: str, save_path: str, debugging: bool = False):
        self._process_spline_data()
        self._calculate_stats()

        if debugging:
            self._plot_results()

        self._save_data_to_csv(csv_path, save_path)

    def _process_spline_data(self):
        for subdir, image_data in self.spline_data.items():
            longest_l = 0.0
            error_found = False

            for img, splines in image_data.items():
                for spline, spline_data in splines.items():
                    if spline_data[1] == 0.0:
                        error_found = True
                        break

                    if spline == "2":
                        l = spline_data[1]
                        self.l_vals[subdir].append(l)

                        if l > longest_l:
                            longest_l = l
                            self.reduced_spline_data[subdir] = splines
                            self.images_used[subdir] = img

                if error_found:
                    break

    def _calculate_stats(self):
        # Calculate average l for each subdir
        for subdir, l_list in self.l_vals.items():
            if l_list:  # Ensure list is not empty
                self.subdir_means[subdir] = np.mean(l_list)

        if not self.subdir_means:
            print("No 'l' values found to calculate statistics or plot.")
            return

        # Calculate global mean and std of subdir averages
        all_means = list(self.subdir_means.values())
        self.global_mean = np.mean(all_means)
        self.global_std = np.std(all_means)

        print(f"Global Mean of Subdir Average 'l': {self.global_mean:.2f}")
        print(f"Global Std Dev of Subdir Average 'l': {self.global_std:.2f}")

        # Mark subdirs based on the condition
        self.threshold = self.global_mean - self.global_std
        for subdir, mean_l in self.subdir_means.items():
            # Using reduced_spline_data's l seems inconsistent with plotting means.
            # Sticking to the user's request to plot means, let's use subdir_means for error check.
            # Original pseudo code check was: if splines['2'][1] > global_mean - std:
            # Using mean instead:
            if mean_l > self.threshold:
                self.subdir_errors[subdir] = False  # No error
            else:
                self.subdir_errors[subdir] = True  # Error

    def _save_data_to_csv(self, path_to_csv: str, save_path: str):
        if not path_to_csv.lower().endswith(".csv"):
            raise ValueError(f"The provided path '{path_to_csv}' is not a CSV file.")

        if not self.subdir_means:
            raise ValueError(
                "No evaluation data available to save. Run calculations first."
            )

        directory = os.path.dirname(path_to_csv)
        if directory and not os.path.exists(directory):
            raise FileNotFoundError(
                f"The directory '{directory}' does not exist. Please create it before saving the CSV."
            )

        if not os.path.exists(path_to_csv):
            raise FileNotFoundError(
                f"The CSV file '{path_to_csv}' does not exist. Cannot add columns."
            )

        df = pd.read_csv(path_to_csv)
        print(f"Successfully read {path_to_csv}. Shape: {df.shape}")

        index_col_name = "index"
        if index_col_name not in df.columns:
            raise ValueError(
                f"Required column '{index_col_name}' not found in {path_to_csv}. Available columns: {df.columns.tolist()}"
            )

        new_data = {"tcd_low": [], "tcd_med": [], "tcd_high": [], "coords": []}

        for idx_val in df[index_col_name]:
            subdir_key = str(idx_val)
            spline_info = self.reduced_spline_data.get(subdir_key)

            # Skip outliers - only add data for non-error subdirectories
            is_outlier = self.subdir_errors.get(
                subdir_key, True
            )  # Default to True if not found

            if spline_info and not is_outlier:
                tcd_low = (
                    spline_info.get("0", [np.nan])[0]
                    if isinstance(spline_info.get("0"), list) and spline_info.get("0")
                    else np.nan
                )
                tcd_med = (
                    spline_info.get("1", [np.nan])[0]
                    if isinstance(spline_info.get("1"), list) and spline_info.get("1")
                    else np.nan
                )
                tcd_high = (
                    spline_info.get("2", [np.nan])[0]
                    if isinstance(spline_info.get("2"), list) and spline_info.get("2")
                    else np.nan
                )

                coords_0 = (
                    spline_info.get("0", [None, None, []])[2]
                    if isinstance(spline_info.get("0"), list)
                    and len(spline_info.get("0")) > 2
                    else []
                )
                coords_1 = (
                    spline_info.get("1", [None, None, []])[2]
                    if isinstance(spline_info.get("1"), list)
                    and len(spline_info.get("1")) > 2
                    else []
                )
                coords_2 = (
                    spline_info.get("2", [None, None, []])[2]
                    if isinstance(spline_info.get("2"), list)
                    and len(spline_info.get("2")) > 2
                    else []
                )

                coords_0 = coords_0 if isinstance(coords_0, list) else []
                coords_1 = coords_1 if isinstance(coords_1, list) else []
                coords_2 = coords_2 if isinstance(coords_2, list) else []

                concatenated_coords = coords_0 + coords_1 + coords_2

                new_data["tcd_low"].append(tcd_low)
                new_data["tcd_med"].append(tcd_med)
                new_data["tcd_high"].append(tcd_high)
                new_data["coords"].append(concatenated_coords)
            else:
                new_data["tcd_low"].append(np.nan)
                new_data["tcd_med"].append(np.nan)
                new_data["tcd_high"].append(np.nan)
                new_data["coords"].append([])

        for col_name, data_list in new_data.items():
            if len(data_list) != len(df):
                raise ValueError(
                    f"Length mismatch for column '{col_name}'. Expected {len(df)}, got {len(data_list)}. Check loop logic."
                )
            df[col_name] = data_list

        df.to_csv(
            save_path, index=False
        )  # Overwrite the original file, don't write pandas index
        print(f"Successfully updated and saved data to {save_path}")

    def _plot_results(self):
        subdirs = list(self.subdir_means.keys())
        subdirs.sort(key=int)
        means = [self.subdir_means[s] for s in subdirs]
        colors = [
            "red" if self.subdir_errors.get(s, False) else "green" for s in subdirs
        ]
        markers = ["x" if s in self.ground_truth else "." for s in subdirs]

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size as needed

        # Keep track of scatter plot artists for mplcursors
        scatter_artists = []
        scatter_labels = []

        for i, subdir in enumerate(subdirs):
            # Create scatter plot for each point individually
            scatter = ax.scatter(
                subdir, means[i], color=colors[i], marker=markers[i], s=100
            )
            scatter_artists.append(scatter)
            image_name = self.images_used.get(subdir, "N/A")
            scatter_labels.append(f"Subdir: {subdir}\nImage: {image_name}")

        # Add hover tooltips using mplcursors
        cursor = mplcursors.cursor(scatter_artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            # Find the index corresponding to the selected artist
            artist_index = scatter_artists.index(sel.artist)
            sel.annotation.set_text(scatter_labels[artist_index])
            sel.annotation.get_bbox_patch().set(
                alpha=0.8
            )  # Optional: make tooltip slightly transparent

        ax.axhline(
            self.global_mean,
            color="blue",
            linestyle="--",
            label=f"Global Mean ({self.global_mean:.2f})",
        )
        ax.axhline(
            self.threshold,
            color="orange",
            linestyle=":",
            label=f"Threshold (Mean - Std) ({self.threshold:.2f})",
        )

        ax.set_xlabel("Subdirectory")
        ax.set_ylabel("Average Length 'l'")
        ax.set_title("Average Spline Length 'l' per Subdirectory")
        plt.xticks(rotation=90, fontsize=8)  # Rotate labels if many subdirs
        ax.legend()
        plt.tight_layout()  # Adjust layout to prevent labels overlapping
        plt.savefig("avg_top_spline_length.png", bbox_inches="tight")
        plt.show()  # Show the interactive plot
        plt.close(fig)
