import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast  # Import ast for literal_eval


class Preprocessing:
    def __init__(self, data_path=None, device=None, df=None, set_seed=False):
        """
        Initialize the preprocessing class for data with coordinate targets.

        This class handles the preprocessing pipeline, including loading data,
        scaling input features, parsing and scaling coordinate target values,
        and splitting data into training/validation/test sets.

        Args:
            data_path (str, optional): Path to the CSV dataset
            device (torch.device, optional): Device to use for computation (CPU or GPU)
            df (pd.DataFrame, optional): DataFrame to use instead of loading from file
            set_seed (bool): Whether to set random seeds for reproducibility
        """
        self.device = device if device is not None else torch.device("cpu")

        if set_seed:
            self._set_random_seeds()

        if df is not None:
            self.df = df.copy()  # Use copy to avoid modifying original DataFrame
            print(f"Using provided DataFrame. Initial shape: {self.df.shape}")
            self._handle_nans()
        elif data_path is not None:
            self.load_data(data_path=data_path)
        else:
            self.df = None

        # Renamed/removed attributes related to TCD
        self.scaled_features = None  # Will store the scaled input features DataFrame
        self.feature_names = None
        self.input_scaling_params = None  # Renamed from scaling_params
        self.coords_scaling_params = None  # Replaces tcd_scaling_params
        self.y = None  # Will store the scaled, flattened coordinates

    def _set_random_seeds(self, seed=42):
        """
        Set random seeds for reproducibility across PyTorch, NumPy, and CUDA if available.

        Args:
            seed (int): Seed value for random number generators

        Returns:
            self: Returns the instance for method chaining
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return self

    def _handle_nans(self):
        """Helper method to handle NaN values in the DataFrame."""
        if self.df is None:
            return
        print("\nNaN counts per column before dropping:")
        print(self.df.isna().sum())
        original_shape = self.df.shape
        self.df.dropna(inplace=True)
        print(f"\nShape after dropping rows with NaNs: {self.df.shape}")
        if self.df.empty:
            print("Warning: DataFrame is empty after dropping NaNs.")
        elif original_shape != self.df.shape:
            print(
                f"Dropped {original_shape[0] - self.df.shape[0]} rows containing NaNs."
            )

    def load_data(self, data_path=None, df=None):
        """
        Load data from a CSV file or a provided DataFrame.

        Handles NaN values after loading.

        Args:
            data_path (str, optional): Path to the CSV dataset
            df (pd.DataFrame, optional): DataFrame to use

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If neither data_path nor df is provided
            FileNotFoundError: If data_path is provided but file doesn't exist
        """
        if df is not None:
            self.df = df.copy()  # Use copy to avoid modifying original DataFrame
            print(f"Using provided DataFrame. Initial shape: {self.df.shape}")
        elif data_path is not None:
            try:
                self.df = pd.read_csv(data_path)
                print(f"Successfully read {data_path}.")
                print("Original DataFrame head:")
                print(self.df.head())
                print(f"Original shape: {self.df.shape}")
            except FileNotFoundError:
                print(f"Error: CSV file not found at {data_path}")
                raise
            except Exception as e:
                print(f"Error reading CSV file {data_path}: {e}")
                raise
        else:
            raise ValueError("Either data_path or df must be provided")

        self._handle_nans()  # Handle NaNs after loading/setting df
        return self

    def parse_array_string(self, array_str):
        """
        Convert string representation of a 2D coordinate array to a numpy array.

        Uses ast.literal_eval for safe parsing of string representations like
        '[[x1, y1], [x2, y2], ...]' into lists, then converts to a numpy array.

        Args:
            array_str (str): String representation of a 2D coordinate array

        Returns:
            np.ndarray: Numpy array of shape (N, 2) with float values, or None if parsing fails.
        """
        try:
            # Safely evaluate the string literal
            parsed_list = ast.literal_eval(array_str)
            # Ensure it's a list of lists/tuples (representing points)
            if isinstance(parsed_list, list) and all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in parsed_list
            ):
                # Convert to float, handling potential non-numeric values gracefully if needed
                try:
                    return np.array(parsed_list, dtype=float)
                except ValueError:
                    print(
                        f"Warning: Could not convert all elements in {array_str} to float."
                    )
                    return None
            else:
                print(
                    f"Warning: Parsed data is not in the expected format (list of pairs): {array_str}"
                )
                return None
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Warning: Could not parse array string: {array_str}. Error: {e}")
            return None  # Return None or raise an error, depending on desired behavior

    def scale_features(
        self,
        columns_to_drop=None,
        save_params=True,
        params_path="neural_network_framework/scaling_data/input_params.json",
    ):
        """
        Scale input features using Z-score normalization and remove unwanted columns.

        Assumes 'coords' is the target column.

        Args:
            columns_to_drop (list, optional): List of additional column names to drop
                                              Defaults to ['index'] if None.
            save_params (bool): Whether to save scaling parameters to a file
            params_path (str): Path to save the input scaling parameters JSON file

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If no data has been loaded or if 'coords' column is missing.
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        if "coords" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'coords' column.")

        # Prepare columns to drop (default + user-specified)
        default_drop = [
            "index",
            "coords",
            "tcd_low",
            "tcd_med",
            "tcd_high",
        ]  # Always drop target and image name from features
        to_drop = set(default_drop)
        if columns_to_drop:
            to_drop.update(columns_to_drop)

        # Ensure columns exist before trying to drop
        actual_to_drop = [col for col in to_drop if col in self.df.columns]
        X = self.df.drop(columns=actual_to_drop)

        # Check for constant columns *after* dropping specified ones
        variances = X.var()
        constant_columns = variances[variances == 0].index.tolist()
        if constant_columns:
            print(f"Dropping constant columns: {constant_columns}")
            X = X.drop(constant_columns, axis=1)

        if X.empty:
            print("Warning: No feature columns remaining after dropping.")
            self.scaled_features = pd.DataFrame()
            self.feature_names = []
            self.input_scaling_params = {"feature_names": [], "mean": {}, "std": {}}
            return self

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.feature_names = X.columns.tolist()
        self.input_scaling_params = {
            "feature_names": self.feature_names,
            "mean": dict(zip(self.feature_names, scaler.mean_)),
            "std": dict(
                zip(self.feature_names, scaler.scale_.tolist())
            ),  # Save std as list
        }

        # Store scaled features
        self.scaled_features = pd.DataFrame(
            X_scaled, columns=self.feature_names, index=X.index
        )  # Preserve index

        if save_params:
            # Ensure directory exists
            import os

            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(self.input_scaling_params, f, indent=4)

        return self

    def process_and_scale_coordinates(
        self,
        save_params=True,
        params_path="neural_network_framework/scaling_data/coords_params.json",
    ):
        """
        Parses, flattens, and scales coordinate data from the 'coords' column.

        Scaling (Z-score) is applied independently to x and y dimensions across the dataset.

        Args:
            save_params (bool): Whether to save scaling parameters to a file.
            params_path (str): Path to save the coordinate scaling parameters JSON file.

        Returns:
            self: Returns the instance for method chaining.

        Raises:
            ValueError: If no data has been loaded or if 'coords' column is missing or invalid.
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")
        if "coords" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'coords' column.")

        # Parse coordinate strings, filter out None results from parsing errors
        parsed_coords = self.df["coords"].apply(self.parse_array_string)
        valid_indices = (
            parsed_coords.notna()
        )  # Get indices where parsing was successful
        parsed_coords = parsed_coords[valid_indices].tolist()
        # Align scaled_features and df with valid coords indices
        self.df = self.df[valid_indices].copy()
        if self.scaled_features is not None:
            self.scaled_features = self.scaled_features.loc[valid_indices].copy()

        if not parsed_coords:
            raise ValueError("No valid coordinate data found after parsing.")

        # Check if all arrays have the same flattened length
        flattened_lengths = [coords.flatten().shape[0] for coords in parsed_coords]
        if len(set(flattened_lengths)) > 1:
            print(
                "Warning: Coordinate arrays have different flattened lengths after parsing."
            )
            # Decide how to handle this: error out, pad, or only use consistent ones?
            # For now, let's proceed but be aware. Scaling might be problematic.
            # A safer approach might be to filter based on a common length.
        target_dim = (
            flattened_lengths[0] if flattened_lengths else 0
        )  # Get expected dimension

        # Collect all x and y coordinates across all valid samples
        all_x = []
        all_y = []
        for coords in parsed_coords:
            if coords.ndim == 2 and coords.shape[1] == 2:
                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])
            # Handle cases where parsing might yield unexpected shapes, if necessary

        if not all_x or not all_y:
            raise ValueError("Could not extract valid x and y coordinates.")

        # Calculate mean and std for x and y separately
        mean_x = np.mean(all_x)
        std_x = np.std(all_x)
        mean_y = np.mean(all_y)
        std_y = np.std(all_y)

        # Handle potential zero std deviation (constant coordinates)
        std_x = std_x if std_x > 1e-6 else 1.0
        std_y = std_y if std_y > 1e-6 else 1.0

        self.coords_scaling_params = {
            "mean_x": mean_x,
            "std_x": std_x,
            "mean_y": mean_y,
            "std_y": std_y,
            "target_dim": target_dim,  # Store the flattened dimension
        }

        if save_params:
            # Ensure directory exists
            import os

            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(self.coords_scaling_params, f, indent=4)

        # Apply scaling and flatten
        scaled_flattened_coords = []
        for coords in parsed_coords:
            if coords.ndim == 2 and coords.shape[1] == 2:
                scaled_coords = np.empty_like(coords, dtype=float)
                scaled_coords[:, 0] = (coords[:, 0] - mean_x) / std_x
                scaled_coords[:, 1] = (coords[:, 1] - mean_y) / std_y
                scaled_flattened_coords.append(scaled_coords.flatten())
            else:
                # Handle inconsistent shapes if necessary, e.g., append NaNs or skip
                # For now, assume consistent shape based on earlier check or filtering
                print(
                    f"Skipping coordinate array with unexpected shape: {coords.shape}"
                )

        self.y = scaled_flattened_coords  # Store as a list of numpy arrays

        return self

    def preprocess(
        self,
        save_params=True,
        input_params_path="neural_network_framework/scaling_data/input_params.json",
        coords_params_path="neural_network_framework/scaling_data/coords_params.json",
    ):
        """
        Run the complete preprocessing pipeline for coordinate data.

        1. scale_features(): Scale input features.
        2. process_and_scale_coordinates(): Parse, flatten, and scale target coordinates.

        Args:
            save_params (bool): Whether to save scaling parameters to files.
            input_params_path (str): Path for saving input feature scaling parameters.
            coords_params_path (str): Path for saving coordinate scaling parameters.

        Returns:
            self: Returns the instance for method chaining.
        """
        # Ensure data is loaded
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")

        self.scale_features(save_params=save_params, params_path=input_params_path)
        self.process_and_scale_coordinates(
            save_params=save_params, params_path=coords_params_path
        )
        # Note: self.scaled_features and self.y now hold the processed data
        # self.df still holds the original (but NaN-dropped) data
        return self

    def get_processed_data(self):
        """
        Return the scaled features and scaled/flattened target coordinates.

        Returns:
            tuple: (pd.DataFrame, list): Scaled features (X) and scaled/flattened coordinates (y)

        Raises:
            ValueError: If preprocess() hasn't been called successfully first.
        """
        if self.scaled_features is None or self.y is None:
            raise ValueError("Preprocessing not complete. Run preprocess() first.")
        return self.scaled_features, self.y

    def split_data(
        self, train_size=0.8, val_size=0.1, random_state=42, return_tensors=True
    ):
        """
        Split the processed data into train, validation, and test sets.

        Args:
            train_size (float): Proportion for training (default: 0.8).
            val_size (float): Proportion for validation (default: 0.1).
            random_state (int): Random seed for reproducibility.
            return_tensors (bool): Return PyTorch tensors (True) or numpy arrays (False).

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)

        Raises:
            ValueError: If preprocess() hasn't run or split proportions are invalid.
        """
        if self.scaled_features is None or self.y is None:
            raise ValueError("Preprocessing not complete. Run preprocess() first.")

        test_size = 1.0 - train_size - val_size
        if not (0 < test_size < 1):  # Ensure test_size is positive and less than 1
            raise ValueError(
                f"Invalid split proportions. Train ({train_size}) + Val ({val_size}) must be less than 1.0, resulting in a positive test size."
            )

        # Get scaled features (X) and scaled/flattened coordinates (y)
        X = self.scaled_features.values
        # Convert list of numpy arrays to a 2D numpy array for splitting
        try:
            y = np.array(self.y)
            # Basic check for consistent shapes before splitting
            if y.ndim != 2:
                raise ValueError(
                    f"Target 'y' data could not be stacked into a consistent 2D array. Check coordinate lengths. Shape: {y.shape}"
                )
        except ValueError as e:
            # This might happen if flattened arrays have different lengths
            raise ValueError(
                f"Could not convert target 'y' (list of arrays) into a single NumPy array for splitting. Check for consistent flattened coordinate lengths. Original error: {e}"
            )

        # Ensure X and y have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatch between number of samples in features ({X.shape[0]}) and targets ({y.shape[0]}) after processing."
            )

        # First split: Train vs. Temp (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state
        )

        # Second split: Val vs. Test from Temp
        # Calculate the proportion of the temp set that should be test data
        relative_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, random_state=random_state
        )

        if return_tensors:
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)

        return X_train, y_train, X_val, y_val, X_test, y_test


# TCDDataset name might be misleading now, consider renaming if used elsewhere
# For now, keep as is, but it works generically with any X, y tensors/arrays
class TCDDataset(Dataset):
    """
    PyTorch Dataset.

    Works generically for (features, target) pairs.

    Attributes:
        X (torch.Tensor): Input features
        y (torch.Tensor): Target values
    """

    def __init__(self, X, y):
        """
        Initialize the dataset.

        Args:
            X (torch.Tensor or np.ndarray): Input features
            y (torch.Tensor or np.ndarray): Target values
        """
        self.X = torch.FloatTensor(X) if not isinstance(X, torch.Tensor) else X
        self.y = torch.FloatTensor(y) if not isinstance(y, torch.Tensor) else y

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (features, target) for the requested sample
        """
        return self.X[idx], self.y[idx]
