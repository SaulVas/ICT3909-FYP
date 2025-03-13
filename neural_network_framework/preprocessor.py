import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessing:
    def __init__(self, data_path=None, device=None, df=None, set_seed=False):
        """
        Initialize the preprocessing class for TCD data.

        This class handles the complete preprocessing pipeline for TCD (Transcranial Doppler) data,
        including loading data, processing string arrays into numerical format, scaling features,
        normalizing target values, and splitting data into training/validation/test sets.

        Args:
            data_path (str, optional): Path to the CSV dataset containing TCD data
            device (torch.device, optional): Device to use for computation (CPU or GPU)
            df (pd.DataFrame, optional): DataFrame to use instead of loading from file
            set_seed (bool): Whether to set random seeds for reproducibility
        """
        self.device = device if device is not None else torch.device("cpu")

        if set_seed:
            self._set_random_seeds()

        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            self.df = None

        self.processed_data = None
        self.feature_names = None
        self.scaling_params = None
        self.tcd_scaling_params = None
        self.y = None

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

    def load_data(self, data_path=None, df=None):
        """
        Load data from a CSV file or a provided DataFrame.

        This method allows loading data after the class has been initialized,
        either from a file path or directly from a DataFrame.

        Args:
            data_path (str, optional): Path to the CSV dataset
            df (pd.DataFrame, optional): DataFrame to use

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If neither data_path nor df is provided
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        return self

    def parse_array_string(self, array_str):
        """
        Convert string representation of array to actual numpy array.

        This helper method parses string representations of arrays in the format "[x1, x2, ...]"
        into actual numpy arrays of float values.

        Args:
            array_str (str): String representation of an array

        Returns:
            np.ndarray: Numpy array of float values
        """
        return np.array([float(x.strip()) for x in array_str.strip("[]").split(",")])

    def process_tcd_data(self):
        """
        Process TCD columns from string representations to numpy arrays.

        This method converts the string representations of TCD arrays in the columns
        'tcd_low', 'tcd_med', and 'tcd_high' into actual numpy arrays, then combines
        them into a single 'tcd_data' column.

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If no data has been loaded
        """
        if self.df is None:
            raise ValueError("No data loaded. Use load_data() first.")

        self.processed_data = self.df.copy()

        self.processed_data["tcd_low"] = self.processed_data["tcd_low"].apply(
            self.parse_array_string
        )
        self.processed_data["tcd_med"] = self.processed_data["tcd_med"].apply(
            self.parse_array_string
        )
        self.processed_data["tcd_high"] = self.processed_data["tcd_high"].apply(
            self.parse_array_string
        )

        self.processed_data["tcd_data"] = self.processed_data.apply(
            lambda row: np.concatenate(
                [row["tcd_low"], row["tcd_med"], row["tcd_high"]]
            ),
            axis=1,
        )

        self.processed_data = self.processed_data.drop(
            ["tcd_low", "tcd_med", "tcd_high"], axis=1
        )

        return self

    def scale_features(
        self, save_params=True, params_path="scaling_data/input_params.json"
    ):
        """
        Scale features using Z-score normalization and remove unwanted columns.

        This method:
        1. Removes the 'image_name' column
        2. Separates features and target data
        3. Removes constant columns (zero variance)
        4. Scales features using StandardScaler (Z-score normalization)
        5. Stores scaling parameters for later use
        6. Optionally saves scaling parameters to a JSON file

        Args:
            save_params (bool): Whether to save scaling parameters to a file
            params_path (str): Path to save the scaling parameters JSON file

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If process_tcd_data() hasn't been called first
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run process_tcd_data() first.")

        self.processed_data = self.processed_data.drop(["image_name"], axis=1)

        X = self.processed_data.drop("tcd_data", axis=1)
        self.y = self.processed_data["tcd_data"].tolist()

        variances = X.var()
        constant_columns = variances[variances == 0].index.tolist()
        if constant_columns:
            X = X.drop(constant_columns, axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.feature_names = X.columns.tolist()
        self.scaling_params = {
            "feature_names": self.feature_names,
            "mean": dict(zip(self.feature_names, scaler.mean_)),
            "std": dict(zip(self.feature_names, scaler.scale_)),
        }

        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        if save_params:
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(self.scaling_params, f, indent=4)

        self.processed_data = X_scaled_df.copy()

        return self

    def scale_tcd_data(
        self, save_params=True, params_path="scaling_data/output_params.json"
    ):
        """
        Scale TCD data using Z-score normalization.

        This method:
        1. Converts the list of TCD arrays to a 2D numpy array
        2. Calculates mean and standard deviation for each dimension
        3. Scales each dimension individually using Z-score normalization
        4. Stores scaling parameters for later use
        5. Optionally saves scaling parameters to a JSON file
        6. Updates the processed_data with scaled TCD values

        Args:
            save_params (bool): Whether to save scaling parameters to a file
            params_path (str): Path to save the TCD scaling parameters JSON file

        Returns:
            self: Returns the instance for method chaining

        Raises:
            ValueError: If scale_features() hasn't been called first
        """
        if self.y is None:
            raise ValueError("No target data. Run scale_features() first.")

        tcd_arrays = np.array(self.y)

        tcd_means = np.mean(tcd_arrays, axis=0)
        tcd_stds = np.std(tcd_arrays, axis=0)

        tcd_arrays_scaled = np.zeros_like(tcd_arrays)
        for i in range(tcd_arrays.shape[1]):
            tcd_arrays_scaled[:, i] = (tcd_arrays[:, i] - tcd_means[i]) / tcd_stds[i]

        self.tcd_scaling_params = {
            "means": tcd_means.tolist(),
            "stds": tcd_stds.tolist(),
        }

        if save_params:
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(self.tcd_scaling_params, f, indent=4)

        self.processed_data["tcd_data"] = [row for row in tcd_arrays_scaled]

        return self

    def preprocess(self, save_params=True):
        """
        Run the complete preprocessing pipeline.

        This method chains together the three main preprocessing steps:
        1. process_tcd_data(): Convert string arrays to numpy arrays
        2. scale_features(): Scale input features and remove unwanted columns
        3. scale_tcd_data(): Scale target TCD data

        Args:
            save_params (bool): Whether to save scaling parameters to files

        Returns:
            self: Returns the instance for method chaining
        """
        return (
            self.process_tcd_data()
            .scale_features(save_params=save_params)
            .scale_tcd_data(save_params=save_params)
        )

    def get_processed_data(self):
        """
        Return the processed data DataFrame.

        Returns:
            pd.DataFrame: The fully processed data

        Raises:
            ValueError: If preprocess() hasn't been called first
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run preprocess() first.")
        return self.processed_data

    def split_data(
        self, train_size=0.8, val_size=0.1, random_state=42, return_tensors=True
    ):
        """
        Split the processed data into train, validation, and test sets.

        This method:
        1. Extracts features and target from processed data
        2. Performs a two-stage split to create train/validation/test sets
        3. Optionally converts the data to PyTorch tensors and moves them to the specified device

        The split is performed in two stages:
        - First split: Separate training data from the rest
        - Second split: Divide the remaining data into validation and test sets

        Args:
            train_size (float): Proportion of data to use for training (default: 0.8)
            val_size (float): Proportion of data to use for validation (default: 0.1)
            random_state (int): Random seed for reproducibility
            return_tensors (bool): Whether to return PyTorch tensors or numpy arrays
                                  Set to True when using with PyTorch models directly
                                  Set to False when additional preprocessing is needed

        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test) in either tensor or numpy format

        Raises:
            ValueError: If preprocess() hasn't been called first or if split proportions are invalid
        """
        if self.processed_data is None:
            raise ValueError("No processed data. Run preprocess() first.")

        test_size = 1.0 - train_size - val_size
        if test_size <= 0:
            raise ValueError(
                f"Invalid split proportions. Train ({train_size}) + Val ({val_size}) must be less than 1.0"
            )

        X = self.processed_data.drop("tcd_data", axis=1).values
        y = np.array([np.array(arr) for arr in self.processed_data["tcd_data"]])

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), random_state=random_state
        )

        temp_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=temp_test_size, random_state=random_state
        )

        if return_tensors:
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            X_test = torch.FloatTensor(X_test).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)

        return X_train, y_train, X_val, y_val, X_test, y_test


class TCDDataset(Dataset):
    """
    PyTorch Dataset for TCD data.

    This class implements the PyTorch Dataset interface for TCD data,
    making it easy to use with DataLoader for batch processing during training.

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
