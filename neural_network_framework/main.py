import torch
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from preprocessor import Preprocessing, TCDDataset
from neural_net import FeedForwardNN
from trainer import Trainer
from hyper_parameter_tuning import run_optuna_study
from visualisation import Visualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

USE_OPTUNA = True
USE_K_FOLD = True
N_TRIALS = 10

# Create visualizer for plotting
visualizer = Visualizer(output_dir="neural_network_framework/plots")

# Check target dimensions before training
preprocessor = Preprocessing("data/on_water_dataset/processed_data.csv", device=device)
preprocessor.preprocess()

X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()

# Print shapes to debug
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

train_dataset = TCDDataset(X_train, y_train)
val_dataset = TCDDataset(X_val, y_val)
test_dataset = TCDDataset(X_test, y_test)

# For k-fold cross-validation, combine train and validation sets
if USE_K_FOLD:
    X_combined = torch.cat([X_train, X_val], dim=0)
    y_combined = torch.cat([y_train, y_val], dim=0)
    combined_dataset = TCDDataset(X_combined, y_combined)
else:
    combined_dataset = None

BATCH_SIZE = 20
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Get input dimension from data
input_dim = X_train.shape[1]
# Get output dimension from target data
output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

# Define whether to use hyperparameter optimization

if USE_OPTUNA:
    print("Running hyperparameter optimization...")
    best_params = run_optuna_study(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        n_trials=N_TRIALS,
        k_fold=USE_K_FOLD,
        dataset=combined_dataset if USE_K_FOLD else None,
        visualizer=visualizer,
    )

    # Create model with best hyperparameters
    model = FeedForwardNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=best_params["hidden_layers"],
        dropout_rates=best_params["dropout_rates"],
    ).to(device)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        visualizer=visualizer,
    )
else:
    # Use default hyperparameters
    hidden_layers = [128, 64, 32]
    dropout_rates = [0.2, 0.2, 0.2]

    model = FeedForwardNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
    ).to(device)

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        visualizer=visualizer,
    )

# Train the model
print("Training final model...")
val_loss, training_time = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    early_stopping_patience=20,
    verbose=True,
)

# Test the model
test_loss = trainer.test(test_loader)
print(f"Test MSE: {test_loss:.6f}")
print(f"Training time: {training_time:.2f} seconds")

# Save the model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(
    model.state_dict(), f"neural_network_framework/models/best_model_{timestamp}.pt"
)
print(f"Model saved to best_model_{timestamp}.pt")

# Print location of saved plots
print(f"All plots have been saved to: {visualizer.output_dir}")
