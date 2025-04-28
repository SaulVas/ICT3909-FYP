import numpy as np
from collections import defaultdict
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

    def __call__(self):
        self._process_spline_data()
        self.calculate_stats_and_plot()

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

    def calculate_stats_and_plot(self):
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
        threshold = self.global_mean - self.global_std
        for subdir, mean_l in self.subdir_means.items():
            # Using reduced_spline_data's l seems inconsistent with plotting means.
            # Sticking to the user's request to plot means, let's use subdir_means for error check.
            # Original pseudo code check was: if splines['2'][1] > global_mean - std:
            # Using mean instead:
            if mean_l > threshold:
                self.subdir_errors[subdir] = False  # No error
            else:
                self.subdir_errors[subdir] = True  # Error

        # Prepare data for plotting
        subdirs = list(self.subdir_means.keys())
        # Sort subdirectories numerically assuming they are strings representing integers
        subdirs.sort(key=int)
        means = [
            self.subdir_means[s] for s in subdirs
        ]  # Ensure means align with sorted subdirs
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
            threshold,
            color="orange",
            linestyle=":",
            label=f"Threshold (Mean - Std) ({threshold:.2f})",
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
