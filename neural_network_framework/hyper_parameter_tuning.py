import optuna
from neural_net import FeedForwardNN
from trainer import Trainer


def objective(
    trial, train_loader, val_loader, input_dim, device, k_fold=False, dataset=None
):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Input dimension of the model
        device: Device to run on (CPU or GPU)
        k_fold: Whether to use k-fold cross-validation
        dataset: Full dataset (required if k_fold=True)

    Returns:
        val_loss: Validation loss (or mean validation loss if using k-fold)
    """
    # Define hyperparameters to optimize
    n_layers = trial.suggest_int("n_layers", 1, 3)

    hidden_layers = []
    dropout_rates = []

    # Define layer sizes and dropout rates
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f"hidden_layer_{i+1}_size", 8, 256))
        dropout_rates.append(trial.suggest_float(f"dropout_rate_{i+1}", 0.0, 0.5))

    # Learning parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    # Create model with the suggested hyperparameters
    model = FeedForwardNN(
        input_dim=input_dim, hidden_layers=hidden_layers, dropout_rates=dropout_rates
    ).to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Train and evaluate
    if k_fold:
        val_loss = trainer.k_fold_cross_validation(
            dataset=dataset,
            n_splits=10,
            batch_size=train_loader.batch_size,
            epochs=200,  # Reduced for faster optimization
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=False,
        )
    else:
        val_loss, _ = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,  # Reduced for faster optimization
            early_stopping_patience=10,
            verbose=False,
        )

    return val_loss


def run_optuna_study(
    train_loader,
    val_loader,
    input_dim,
    device,
    n_trials=1000,
    k_fold=False,
    dataset=None,
):
    """
    Run an Optuna study to find the best hyperparameters.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Input dimension of the model
        device: Device to run on (CPU or GPU)
        n_trials: Number of trials to run
        k_fold: Whether to use k-fold cross-validation
        dataset: Full dataset (required if k_fold=True)

    Returns:
        best_params: Dictionary of best hyperparameters
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, input_dim, device, k_fold, dataset
        ),
        n_trials=n_trials,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.6f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Extract best parameters in a more usable format
    best_params = trial.params

    # Restructure the parameters
    n_layers = best_params["n_layers"]
    hidden_layers = [best_params[f"hidden_layer_{i+1}_size"] for i in range(n_layers)]
    dropout_rates = [best_params[f"dropout_rate_{i+1}"] for i in range(n_layers)]

    return {
        "hidden_layers": hidden_layers,
        "dropout_rates": dropout_rates,
        "learning_rate": best_params["learning_rate"],
        "weight_decay": best_params["weight_decay"],
    }
