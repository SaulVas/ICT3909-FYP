import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap functionality
from Extractor import Extractor
from Detector import Detector
from pathlib import Path


def run_extractor():
    # Define the dataset path
    DATASET_PATH = Path("data/on_water_dataset")  # Use Path object directly

    # Check if the directory exists
    if not DATASET_PATH.is_dir():  # Check if it's a directory
        raise FileNotFoundError(
            f"Dataset directory '{DATASET_PATH}' not found or is not a directory"
        )

    detector = Detector("outputs")
    extractor = Extractor()

    spline_data = {}
    # Iterate through each item in the dataset path
    for subdir_path in DATASET_PATH.iterdir():
        # Check if the item is a directory
        if subdir_path.is_dir():
            print(f"--- Processing subdirectory: {subdir_path.name} ---")

            # Find image files within this specific subdirectory (not recursive)
            image_files_in_subdir = []
            subdir_spline_data = {}
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                # Use glob on the specific subdirectory path
                image_files_in_subdir.extend(str(p) for p in subdir_path.glob(ext))

            if not image_files_in_subdir:
                print(f"No image files found in '{subdir_path.name}'")
                continue  # Move to the next subdirectory

            print(f"Found {len(image_files_in_subdir)} images in '{subdir_path.name}'")

            for image_path in image_files_in_subdir:
                splines = detector(image_path, debug_saves=True)
                tcd_data = extractor(splines, image_path, debugging=True)
                subdir_spline_data[image_path] = tcd_data

            spline_data[subdir_path.name] = subdir_spline_data

    print("--- All subdirectories processed ---")

    with open("computer_vision_pipeline/spline_data.json", "w", encoding="utf-8") as f:
        json.dump(spline_data, f, indent=2)


def evaluate_pipeline(spline_data, ground_truth):

    # Store per-subdirectory statistics
    # Structure: { subdir_key: { spline_idx: {'avg': ..., 'var': ..., 'count': ...}, ... }, ... }
    subdir_stats = {}
    all_subdir_keys = sorted(spline_data.keys())  # For consistent color mapping
    # Add collection for all valid lengths globally
    global_lengths = {"0": [], "1": [], "2": []}

    print(
        "\n--- Calculating Per-Subdirectory Statistics & Collecting Global Lengths ---"
    )
    for subdir_key, images_data in spline_data.items():
        # Collect lengths per spline index *within this subdirectory*
        subdir_lengths = {"0": [], "1": [], "2": []}
        exclude_this_subdir = False  # Flag to exclude subdir if L=0 is found

        if not isinstance(images_data, dict):
            print(
                f"Warning: Unexpected data format for subdir '{subdir_key}'. Skipping subdir."
            )
            continue

        for image_path, splines in images_data.items():
            if exclude_this_subdir:
                break

            if not isinstance(splines, dict):
                print(
                    f"Warning: Unexpected data format for image '{image_path}' in subdir '{subdir_key}'. Skipping image."
                )
                continue

            for spline_key, spline_info in splines.items():
                if exclude_this_subdir:
                    break

                if spline_key not in ["0", "1", "2"]:
                    continue

                if not isinstance(spline_info, (list, tuple)) or len(spline_info) < 2:
                    print(
                        f"Warning: Unexpected data format for spline '{spline_key}' in image '{image_path}'. Skipping spline."
                    )
                    continue

                length_L = spline_info[1]
                if not isinstance(length_L, (int, float)):
                    print(
                        f"Warning: Non-numeric length for spline '{spline_key}' in image '{image_path}'. Skipping spline."
                    )
                    continue

                # Check for L=0 to exclude the entire subdirectory
                if length_L == 0.0:
                    print(
                        f"Info: Found L=0 in subdir '{subdir_key}' (image '{image_path}', spline '{spline_key}'). Excluding this subdir."
                    )
                    exclude_this_subdir = True
                    break
                elif length_L > 0.0:
                    # Collect for subdir stats
                    subdir_lengths[spline_key].append(length_L)
                    # Also collect for global stats if not excluding
                    # (We'll add to global_lengths *after* confirming the subdir isn't excluded)
            # End of spline loop
        # End of image loop

        # --- Subdirectory Processing ---
        if not exclude_this_subdir:
            current_subdir_stat = {}
            has_valid_data_in_subdir = False
            temp_global_lengths = {
                "0": [],
                "1": [],
                "2": [],
            }  # Temp store before adding globally

            for key in ["0", "1", "2"]:
                lengths = subdir_lengths[key]
                count = len(lengths)
                if count > 0:
                    # Collect valid lengths for global calculation from this included subdir
                    temp_global_lengths[key].extend(lengths)

                    avg_l = np.mean(lengths)
                    var_l = np.var(lengths) if count > 1 else np.nan
                    std_l = np.sqrt(var_l) if not np.isnan(var_l) else np.nan
                    current_subdir_stat[key] = {
                        "avg": avg_l,
                        "var": var_l,
                        "std": std_l,
                        "count": count,
                    }
                    has_valid_data_in_subdir = True

            if has_valid_data_in_subdir:
                subdir_stats[subdir_key] = current_subdir_stat
                # Add the collected lengths from this valid subdir to the global pool
                for key in ["0", "1", "2"]:
                    global_lengths[key].extend(temp_global_lengths[key])

    print(f"Processed {len(subdir_stats)} subdirectories with valid data.")

    # --- Calculate Global Statistics (from all individual valid lengths) ---
    final_global_stats = {"avg": {}, "std": {}}
    print("\n--- Global Statistics (Based on ALL valid individual lengths) ---")
    for key in ["0", "1", "2"]:
        lengths = global_lengths[key]
        if lengths:
            global_avg = np.mean(lengths)
            global_std = np.std(lengths)  # Use np.std for population standard deviation
            final_global_stats["avg"][key] = global_avg
            final_global_stats["std"][key] = global_std
            print(
                f"Spline {key}: Global Avg L = {global_avg:.2f}, Global Std Dev L = {global_std:.2f} (from {len(lengths)} total points)"
            )
        else:
            final_global_stats["avg"][key] = np.nan
            final_global_stats["std"][key] = np.nan
            print(f"Spline {key}: No valid data points found globally.")

    # --- Plotting ---
    if not subdir_stats:
        print("No per-subdirectory statistics available to plot.")
        return

    # Prepare data for plotting (Subdirectory stats)
    try:
        sorted_subdir_keys = sorted(subdir_stats.keys(), key=int)
    except ValueError:
        print(
            "Warning: Subdirectory keys are not purely numeric. Sorting alphabetically."
        )
        sorted_subdir_keys = sorted(subdir_stats.keys())

    plot_data = {
        "subdir_keys": sorted_subdir_keys,
        "avg": {"0": [], "1": [], "2": []},
        # 'var': {"0": [], "1": [], "2": []}, # Variance no longer needed for plot
        "std": {"0": [], "1": [], "2": []},
    }

    for key in sorted_subdir_keys:
        stats = subdir_stats[key]
        for spline_idx in ["0", "1", "2"]:
            if spline_idx in stats:
                plot_data["avg"][spline_idx].append(stats[spline_idx]["avg"])
                # plot_data['var'][spline_idx].append(stats[spline_idx]['var']) # Removed
                plot_data["std"][spline_idx].append(stats[spline_idx]["std"])
            else:
                plot_data["avg"][spline_idx].append(np.nan)
                # plot_data['var'][spline_idx].append(np.nan) # Removed
                plot_data["std"][spline_idx].append(np.nan)

    # --- Create Plot: Average L per Spline with Global Stats & Ground Truth Highlighting ---
    fig_avg, axs_avg = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_avg.suptitle(
        "Average Spline Length (L) per Subdirectory with Global Stats", fontsize=16
    )

    legend_handles = []

    for i, spline_idx in enumerate(["0", "1", "2"]):
        ax = axs_avg[i]
        valid_indices = [
            idx
            for idx, val in enumerate(plot_data["avg"][spline_idx])
            if not np.isnan(val)
        ]

        if not valid_indices:
            ax.set_title(f"Spline {spline_idx} (No Data)")
            continue

        # Prepare data for this subplot
        all_x_coords = range(len(valid_indices))
        all_y_values = [plot_data["avg"][spline_idx][idx] for idx in valid_indices]
        all_y_errors = [plot_data["std"][spline_idx][idx] for idx in valid_indices]
        all_y_errors = [err if not np.isnan(err) else 0 for err in all_y_errors]
        all_subdir_labels = [plot_data["subdir_keys"][idx] for idx in valid_indices]

        # Separate data into ground truth (gt) and non-ground truth (ngt)
        x_coords_gt, y_values_gt, y_errors_gt = [], [], []
        x_coords_ngt, y_values_ngt, y_errors_ngt = [], [], []
        tick_labels_gt, tick_labels_ngt = (
            [],
            [],
        )  # Store corresponding labels if needed later
        original_indices_gt, original_indices_ngt = (
            [],
            [],
        )  # Map back to original x-axis positions

        for idx, sub_key in enumerate(all_subdir_labels):
            original_x = all_x_coords[idx]
            if sub_key in ground_truth:
                x_coords_gt.append(original_x)
                y_values_gt.append(all_y_values[idx])
                y_errors_gt.append(all_y_errors[idx])
                tick_labels_gt.append(sub_key)
                original_indices_gt.append(idx)
            else:
                x_coords_ngt.append(original_x)
                y_values_ngt.append(all_y_values[idx])
                y_errors_ngt.append(all_y_errors[idx])
                tick_labels_ngt.append(sub_key)
                original_indices_ngt.append(idx)

        # Plot non-ground truth points (blue)
        if x_coords_ngt:
            ebar_ngt = ax.errorbar(
                x_coords_ngt,
                y_values_ngt,
                yerr=y_errors_ngt,
                fmt="o",
                color="blue",
                markersize=4,
                capsize=3,
                alpha=0.6,
                label="Subdir Avg L \u00b1 Std Dev" if i == 0 else "",
                zorder=2,
            )
            if i == 0:
                legend_handles.append(ebar_ngt)

        # Plot ground truth points (red)
        if x_coords_gt:
            ebar_gt = ax.errorbar(
                x_coords_gt,
                y_values_gt,
                yerr=y_errors_gt,
                fmt="o",
                color="red",
                markersize=5,
                capsize=3,
                alpha=0.9,
                label=(
                    "Ground Truth Subdir"
                    if i == 0
                    and not any(
                        h.get_label() == "Ground Truth Subdir" for h in legend_handles
                    )
                    else ""
                ),
                zorder=3,
            )
            # Add legend handle only once across all subplots
            if i == 0 and not any(
                h.get_label() == "Ground Truth Subdir" for h in legend_handles
            ):
                legend_handles.append(ebar_gt)

        ax.set_title(f"Spline {spline_idx}")
        ax.set_ylabel("Average L")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        # Set ticks and labels based on the combined original x-coords
        ax.set_xticks(all_x_coords)
        ax.set_xticklabels(all_subdir_labels, rotation=90, fontsize=8)

        # Add Global Average and Std Dev lines
        global_avg = final_global_stats["avg"][spline_idx]
        global_std = final_global_stats["std"][spline_idx]

        if not np.isnan(global_avg):
            line_avg = ax.axhline(
                y=global_avg,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Global Avg L" if i == 0 else "",
                zorder=1,
            )
            if i == 0:
                legend_handles.append(line_avg)

            if not np.isnan(global_std):
                line_std_plus = ax.axhline(
                    y=global_avg + global_std,
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    label="Global Avg \u00b1 Global Std Dev" if i == 0 else "",
                    zorder=1,
                )
                ax.axhline(
                    y=global_avg - global_std,
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    zorder=1,
                )
                if i == 0:
                    legend_handles.append(line_std_plus)

    # Add legend to the first subplot
    if legend_handles:
        axs_avg[0].legend(handles=legend_handles, loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save Figure
    plot_filename = "average_l_per_spline_with_global_stats_gt.png"  # Updated filename
    try:
        plt.savefig(plot_filename, bbox_inches="tight")
        print(f"\nSaved plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig_avg)


if __name__ == "__main__":
    output_file_path = Path("computer_vision_pipeline/spline_data.json")
    if not output_file_path.exists():
        print(f"'{output_file_path}' not found. Running extractor...")
        run_extractor()
    else:
        print(f"'{output_file_path}' found. Evaluating existing data...")

    try:
        with open(
            "computer_vision_pipeline/spline_data.json", "r", encoding="utf-8"
        ) as f:
            spline_data = json.load(f)
    except FileNotFoundError:
        print("spline_data.json not found. Run the extractor first.")
    except json.JSONDecodeError:
        print("Error decoding spline_data.json. It might be corrupted.")

    # --- Load Ground Truth Data ---
    ground_truth_file = "computer_vision_pipeline/ground_truth.json"
    ground_truth_set = set()
    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # *** Access the list via the "ground_truth" key ***
            if (
                isinstance(data, dict)
                and "ground_truth" in data
                and isinstance(data["ground_truth"], list)
            ):
                ground_truth_list = data["ground_truth"]
                ground_truth_set = set(str(item) for item in ground_truth_list)
                print(
                    f"Loaded {len(ground_truth_set)} ground truth keys from {ground_truth_file}"
                )
            else:
                print(
                    f"Warning: {ground_truth_file} does not contain a 'ground_truth' key with a list value. Ground truth highlighting disabled."
                )
    except FileNotFoundError:
        print(
            f"Warning: {ground_truth_file} not found. Ground truth highlighting disabled."
        )
    except json.JSONDecodeError:
        print(
            f"Warning: Error decoding {ground_truth_file}. Ground truth highlighting disabled."
        )
    except Exception as e:
        print(
            f"Warning: An unexpected error occurred loading {ground_truth_file}: {e}. Ground truth highlighting disabled."
        )

    evaluate_pipeline(spline_data=spline_data, ground_truth=ground_truth_set)
