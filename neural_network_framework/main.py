import torch
from torch.utils.data import DataLoader
import numpy as np
from preprocessor import Preprocessing, TCDDataset
from visualisation import Visualizer
from ensemble import StackedEnsemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create visualizer for plotting
visualizer = Visualizer(output_dir="neural_network_framework/plots")

# Check target dimensions before training
preprocessor = Preprocessing("data/on_water_dataset/processed_data.csv", device=device)
preprocessor.preprocess()

X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()

X_train_val = np.vstack([X_train, X_val])
y_train_val = np.vstack([y_train, y_val])

# Create datasets and dataloaders
train_dataset = TCDDataset(X_train, y_train)
val_dataset = TCDDataset(X_val, y_val)
test_dataset = TCDDataset(X_test, y_test)

BATCH_SIZE = 20
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Get input and output dimensions
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

print(f"Input dimension: {input_dim}")
print(f"Output dimension: {output_dim}")

# Initialize and train the stacked ensemble
ensemble = StackedEnsemble(
    input_dim=input_dim,
    output_dim=output_dim,
    device=device,
    n_folds=10,
    random_state=42,
    visualizer=visualizer,
)

# Step 1: Tune hyperparameters and save best models
print("Step 1: Tuning hyperparameters...")
best_rf_params, best_ffnn_params = ensemble.tune_hyperparameters(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    train_loader=train_loader,
    val_loader=val_loader,
    n_trials=100,
)

print("\nBest RF parameters:")
print(best_rf_params)
print("\nBest FFNN parameters:")
print(best_ffnn_params)

# Step 2: Generate OOF predictions
print("\nStep 2: Generating out-of-fold predictions...")
rf_oof_preds, ffnn_oof_preds = ensemble.generate_oof_predictions(
    X=X_train_val,
    y=y_train_val,
    train_loader=train_loader,
)

# Step 3: Train meta-learner
print("\nStep 3: Training meta-learner...")
ensemble.train_meta_learner(
    rf_oof_preds=rf_oof_preds,
    ffnn_oof_preds=ffnn_oof_preds,
    y=y_train_val,
)

# Step 4: Evaluate on test set
print("\nStep 4: Evaluating on test set...")

# Get predictions from all models
ensemble_preds = ensemble.predict(X_test)

# Get RF predictions
rf_preds = ensemble.rf_model.predict(X_test)


# Get FFNN predictions
ensemble.ffnn_model.eval()
with torch.no_grad():
    ffnn_preds = ensemble.ffnn_model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

# Calculate metrics for each model
ensemble_mse = mean_squared_error(y_test, ensemble_preds)
ensemble_mae = mean_absolute_error(y_test, ensemble_preds)

rf_mse = mean_squared_error(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)

ffnn_mse = mean_squared_error(y_test, ffnn_preds)
ffnn_mae = mean_absolute_error(y_test, ffnn_preds)

print("\nTest Set Results:")
print(f"Ensemble - MSE: {ensemble_mse:.6f}, MAE: {ensemble_mae:.6f}")
print(f"Random Forest - MSE: {rf_mse:.6f}, MAE: {rf_mae:.6f}")
print(f"FFNN - MSE: {ffnn_mse:.6f}, MAE: {ffnn_mae:.6f}")

# Save meta-weights
print("\nSaving meta-weights...")
ensemble.save_models()

print("\nTraining complete!")
print(f"All plots have been saved to: {visualizer.output_dir}")
