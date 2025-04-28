# pylint: disable=redefined-outer-name broad-exception-caught

import json
from pathlib import Path
from Extractor import Extractor
from Detector import Detector
from Evaluator import Evaluator


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

    evaluator = Evaluator(spline_data, ground_truth_set)

    evaluator(
        csv_path="data/on_water_dataset/collected_data.csv",
        save_path="data/on_water_dataset/processed_data.csv",
    )
