import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
import time


class Trainer:
    def __init__(
        self, model, device, learning_rate=0.001, weight_decay=0, visualizer=None
    ):
        """
        Trainer class to handle training, validation, and testing of neural networks.

        Args:
            model: The neural network model
            device: The device to run computations on (CPU or GPU)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            visualizer: Visualizer instance for plotting (optional)
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.visualizer = visualizer

    def _check_tensor(self, data, name="data"):
        """Ensure data is a PyTorch tensor on the correct device"""
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor, got {type(data)}")
        return data.to(self.device)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Ensure inputs and targets are tensors
            inputs = self._check_tensor(inputs, "inputs")
            targets = self._check_tensor(targets, "targets")

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Ensure inputs and targets are tensors
                inputs = self._check_tensor(inputs, "inputs")
                targets = self._check_tensor(targets, "targets")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        return val_loss

    def train(
        self,
        train_loader,
        val_loader,
        epochs=2000,
        early_stopping_patience=30,
        verbose=True,
    ):
        """
        Train the model with early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            verbose: Whether to print progress

        Returns:
            best_val_loss: Best validation loss achieved
            training_time: Time taken for training
        """
        start_time = time.time()
        best_val_loss = float("inf")
        patience_counter = 0

        # Track losses for plotting
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        training_time = time.time() - start_time

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pt"))

        # Save training history plot if visualizer is available
        if self.visualizer is not None:
            self.visualizer.save_training_history(train_losses, val_losses, epochs)

        return best_val_loss, training_time

    def test(self, test_loader):
        """
        Test the model on the test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            test_loss: Test loss
        """
        self.model.eval()
        test_loss = 0.0

        # For visualization
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = self._check_tensor(inputs, "inputs")
                targets = self._check_tensor(targets, "targets")

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                # Store for visualization
                all_targets.append(targets)
                all_outputs.append(outputs)

        # Save model performance plots if visualizer is available
        if self.visualizer is not None and len(all_targets) > 0:
            all_targets = torch.cat(all_targets, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            self.visualizer.save_model_performance(all_targets, all_outputs, "test")

        return test_loss

    def k_fold_cross_validation(
        self,
        dataset,
        n_splits=5,
        batch_size=20,
        epochs=2000,
        learning_rate=0.001,
        weight_decay=0,
        verbose=True,
    ):
        """
        Perform k-fold cross-validation.

        Args:
            dataset: The full dataset
            n_splits: Number of folds
            batch_size: Batch size for training
            epochs: Number of epochs per fold
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            verbose: Whether to print progress

        Returns:
            mean_val_loss: Mean validation loss across all folds
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_val_losses = []

        X = dataset.X
        y = dataset.y

        # Ensure X and y are tensors
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Dataset X must be a PyTorch tensor, got {type(X)}")
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Dataset y must be a PyTorch tensor, got {type(y)}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if verbose:
                print(f"Fold {fold+1}/{n_splits}")

            # Reset model weights
            for layer in self.model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

            # Create data loaders for this fold
            from torch.utils.data import TensorDataset, DataLoader

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            val_dataset = TensorDataset(X_val_fold, y_val_fold)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

            # Update optimizer with new learning rate and weight decay
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

            # Train on this fold
            val_loss, _ = self.train(
                train_loader, val_loader, epochs=epochs, verbose=verbose
            )
            fold_val_losses.append(val_loss)

            if verbose:
                print(f"Fold {fold+1} validation loss: {val_loss:.6f}")

        mean_val_loss = np.mean(fold_val_losses)
        if verbose:
            print(f"Mean validation loss across {n_splits} folds: {mean_val_loss:.6f}")

        return mean_val_loss
