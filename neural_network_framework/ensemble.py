import json
from typing import Dict, Tuple
from datetime import datetime
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from neural_net import FeedForwardNN
from trainer import Trainer
from hyper_parameter_tuning import run_optuna_study
import optuna
import joblib
from preprocessor import TCDDataset


class StackedEnsemble:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        n_folds: int = 10,
        random_state: int = 42,
        visualizer=None,
    ):
        """
        Initialize the stacked ensemble model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (180 for this case)
            device: Device to run computations on (CPU or GPU)
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            visualizer: Visualizer instance for plotting (optional)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.n_folds = n_folds
        self.random_state = random_state
        self.visualizer = visualizer

        # Initialize models
        self.rf_model = None
        self.ffnn_model = None
        self.meta_weights = None

        # Store best hyperparameters
        self.best_rf_params = None
        self.best_ffnn_params = None

    def _objective_rf(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        num_splits: int = 10,
    ) -> float:
        """
        Optuna objective function for Random Forest hyperparameter tuning using 5-fold CV.

        Args:
            trial: Optuna trial object
            X: Input features (combined train+val)
            y: Target values (combined train+val)

        Returns:
            mean_val_loss: Average validation loss across 5 folds
        """
        # Define hyperparameters to optimize
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        }

        # Perform 5-fold cross-validation
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=self.random_state)
        val_losses = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train RF model
            rf = RandomForestRegressor(**params, random_state=self.random_state)
            rf.fit(X_train, y_train)

            # Predict and compute loss on validation fold
            y_pred = rf.predict(X_val)
            val_loss = mean_squared_error(y_val, y_pred)
            val_losses.append(val_loss)

        # Return average validation loss across all folds
        return np.mean(val_losses)

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_trials: int = 1000,
    ) -> Tuple[Dict, Dict]:
        """
        Tune hyperparameters for both base models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            train_loader: DataLoader for FFNN training
            val_loader: DataLoader for FFNN validation
            n_trials: Number of optimization trials

        Returns:
            Tuple of (best_rf_params, best_ffnn_params)
        """
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.vstack([y_train, y_val])
        dataset = TCDDataset(X_train_val, y_train_val)

        print("Tuning Random Forest hyperparameters...")
        rf_study = optuna.create_study(direction="minimize")
        rf_study.optimize(
            lambda trial: self._objective_rf(trial, X_train_val, y_train_val),
            n_trials=n_trials,
        )
        self.best_rf_params = rf_study.best_params

        # Save best RF model
        best_rf = RandomForestRegressor(
            **self.best_rf_params, random_state=self.random_state
        )
        best_rf.fit(X_train_val, y_train_val)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(
            best_rf, f"neural_network_framework/models/best_rf_{timestamp}.joblib"
        )
        self.rf_model = best_rf

        print("Tuning FFNN hyperparameters...")
        self.best_ffnn_params = run_optuna_study(
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            device=self.device,
            n_trials=n_trials,
            k_fold=True,
            dataset=dataset,
            visualizer=self.visualizer,
        )

        # Create and save best FFNN model
        best_ffnn = FeedForwardNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.best_ffnn_params["hidden_layers"],
            dropout_rates=self.best_ffnn_params["dropout_rates"],
        ).to(self.device)

        trainer = Trainer(
            model=best_ffnn,
            device=self.device,
            learning_rate=self.best_ffnn_params["learning_rate"],
            weight_decay=self.best_ffnn_params["weight_decay"],
            visualizer=self.visualizer,
        )

        # Train best FFNN model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,
            early_stopping_patience=10,
            verbose=True,
        )

        # Save best FFNN model
        torch.save(
            best_ffnn.state_dict(),
            f"neural_network_framework/models/best_ffnn_{timestamp}.pt",
        )
        self.ffnn_model = best_ffnn

        return self.best_rf_params, self.best_ffnn_params

    def generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_loader: torch.utils.data.DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions for both base models.
        For FFNN, implements nested cross-validation with:
        - Outer loop: n_folds for OOF predictions
        - Inner loop: Split training data to have separate validation set for early stopping

        Args:
            X: Input features
            y: Target values
            train_loader: DataLoader for FFNN training

        Returns:
            Tuple of (rf_oof_preds, ffnn_oof_preds)
        """
        kf_outer = KFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        rf_oof_preds = np.zeros_like(y)
        ffnn_oof_preds = np.zeros_like(y)

        for fold, (train_idx, oof_idx) in enumerate(kf_outer.split(X)):
            print(f"Generating OOF predictions for fold {fold + 1}/{self.n_folds}")

            # Split data for outer fold
            X_train_outer, X_oof = X[train_idx], X[oof_idx]
            y_train_outer, _ = y[train_idx], y[oof_idx]

            # Train RF model on all training data from outer fold
            rf = RandomForestRegressor(
                **self.best_rf_params, random_state=self.random_state
            )
            rf.fit(X_train_outer, y_train_outer)

            # Generate OOF predictions for RF
            rf_oof_preds[oof_idx] = rf.predict(X_oof)

            # For FFNN, create inner fold for validation (early stopping)
            # Use n_inner_folds=9 for the inner loop (8:1 train-val split)
            n_inner_folds = 9
            kf_inner = KFold(
                n_splits=n_inner_folds, shuffle=True, random_state=self.random_state
            )

            # Get one split for inner validation
            inner_train_idx, inner_val_idx = next(kf_inner.split(X_train_outer))

            X_train_inner = X_train_outer[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]

            # Train FFNN model with inner validation split
            ffnn = FeedForwardNN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_layers=self.best_ffnn_params["hidden_layers"],
                dropout_rates=self.best_ffnn_params["dropout_rates"],
            ).to(self.device)

            trainer = Trainer(
                model=ffnn,
                device=self.device,
                learning_rate=self.best_ffnn_params["learning_rate"],
                weight_decay=self.best_ffnn_params["weight_decay"],
                visualizer=self.visualizer,
            )

            # Create data loaders for inner fold using TCDDataset
            train_dataset = TCDDataset(X_train_inner, y_train_inner)
            val_dataset = TCDDataset(X_val_inner, y_val_inner)

            inner_train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_loader.batch_size, shuffle=True
            )
            inner_val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=len(val_dataset)
            )

            # Train FFNN with early stopping on inner validation set
            trainer.train(
                train_loader=inner_train_loader,
                val_loader=inner_val_loader,
                epochs=200,
                early_stopping_patience=10,
                verbose=False,
            )

            # Generate OOF predictions for FFNN using the true OOF data
            ffnn.eval()
            with torch.no_grad():
                ffnn_oof_preds[oof_idx] = (
                    ffnn(torch.FloatTensor(X_oof).to(self.device)).cpu().numpy()
                )

        return rf_oof_preds, ffnn_oof_preds

    def train_meta_learner(
        self, rf_oof_preds: np.ndarray, ffnn_oof_preds: np.ndarray, y: np.ndarray
    ) -> None:
        """
        Train the meta-learner using validation-weighted mean approach.

        Args:
            rf_oof_preds: Random Forest OOF predictions
            ffnn_oof_preds: FFNN OOF predictions
            y: True target values
        """
        # Calculate MSE for each output dimension
        rf_mse = np.mean((rf_oof_preds - y) ** 2, axis=0)
        ffnn_mse = np.mean((ffnn_oof_preds - y) ** 2, axis=0)

        # Calculate weights using inverse MSE
        rf_weights = 1 / rf_mse
        ffnn_weights = 1 / ffnn_mse
        total_weights = rf_weights + ffnn_weights

        # Normalize weights
        self.meta_weights = {
            "rf": rf_weights / total_weights,
            "ffnn": ffnn_weights / total_weights,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacked ensemble.

        Args:
            X: Input features

        Returns:
            Ensemble predictions
        """
        # Get base model predictions
        rf_preds = self.rf_model.predict(X)
        self.ffnn_model.eval()
        with torch.no_grad():
            ffnn_preds = (
                self.ffnn_model(torch.FloatTensor(X).to(self.device)).cpu().numpy()
            )

        # Combine predictions using meta-weights
        ensemble_preds = (
            self.meta_weights["rf"] * rf_preds + self.meta_weights["ffnn"] * ffnn_preds
        )

        return ensemble_preds

    def save_models(self, save_dir: str = "neural_network_framework/models") -> None:
        """
        Save the trained models and meta-weights.

        Args:
            save_dir: Directory to save models
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save meta-weights
        with open(
            f"{save_dir}/meta_weights_{timestamp}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "rf": self.meta_weights["rf"].tolist(),
                    "ffnn": self.meta_weights["ffnn"].tolist(),
                },
                f,
                indent=4,
            )

    def load_models(
        self,
        rf_path: str,
        ffnn_path: str,
        meta_weights_path: str,
    ) -> None:
        """
        Load trained models and meta-weights.

        Args:
            rf_path: Path to saved RF model
            ffnn_path: Path to saved FFNN model
            meta_weights_path: Path to saved meta-weights
        """
        # Load RF model
        self.rf_model = joblib.load(rf_path)

        # Load FFNN model
        self.ffnn_model = FeedForwardNN(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_layers=self.best_ffnn_params["hidden_layers"],
            dropout_rates=self.best_ffnn_params["dropout_rates"],
        ).to(self.device)
        self.ffnn_model.load_state_dict(torch.load(ffnn_path))

        # Load meta-weights
        with open(meta_weights_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
            self.meta_weights = {
                "rf": np.array(weights["rf"]),
                "ffnn": np.array(weights["ffnn"]),
            }
