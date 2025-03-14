import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
import seaborn as sns


# pylint: disable=bare-except


class Visualizer:
    def __init__(self, output_dir="plots"):
        """
        Initialize the visualizer with an output directory.

        Args:
            output_dir: Directory to save plots
        """
        # Create timestamp for unique folder names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{output_dir}/{timestamp}"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")

    def save_training_history(self, train_losses, val_losses, epochs_ran):
        """
        Plot and save training and validation loss history.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            epochs_ran: Number of epochs the model was trained for
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, "b-", label="Training Loss")
        plt.plot(epochs, val_losses, "r-", label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Add early stopping indicator
        if len(train_losses) < epochs_ran:
            plt.axvline(
                x=len(train_losses),
                color="g",
                linestyle="--",
                label=f"Early Stopping at Epoch {len(train_losses)}",
            )
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_history.png", dpi=300)
        plt.close()

        # Save raw data for future reference
        history_df = pd.DataFrame(
            {"epoch": epochs, "train_loss": train_losses, "val_loss": val_losses}
        )
        history_df.to_csv(f"{self.output_dir}/training_history.csv", index=False)

        print(f"Training history plot saved to {self.output_dir}/training_history.png")

    def save_optuna_visualizations(self, study):
        """
        Create and save Optuna visualization plots.

        Args:
            study: Completed Optuna study object
        """
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/optuna_history.png", dpi=300)
        plt.close()

        # Plot parameter importances
        try:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/optuna_param_importance.png", dpi=300)
            plt.close()
        except:
            print("Could not generate parameter importance plot (may need more trials)")

        # Plot parallel coordinate plot
        try:
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/optuna_parallel_coordinate.png", dpi=300)
            plt.close()
        except:
            print("Could not generate parallel coordinate plot")

        # NEW: Create detailed hyperparameter exploration plots

        # 1. Extract all trials data into a DataFrame for easier analysis
        trials_df = self._create_trials_dataframe(study)
        trials_df.to_csv(f"{self.output_dir}/all_trials.csv", index=False)

        # 2. Create scatter plots for each hyperparameter vs objective value
        self._plot_hyperparameter_scatter_plots(trials_df)

        # 3. Create distribution plots for each hyperparameter
        self._plot_hyperparameter_distributions(trials_df)

        # 4. Create correlation heatmap between hyperparameters
        self._plot_hyperparameter_correlations(trials_df)

        # 5. Create pairwise scatter plots for most important parameters
        self._plot_pairwise_relationships(trials_df, study)

        # 6. Create interactive HTML visualization if plotly is available
        try:
            self._create_interactive_visualization(study)
        except:
            print(
                "Could not create interactive visualization (plotly may not be installed)"
            )

        # Save best trial parameters
        best_params = study.best_params
        with open(f"{self.output_dir}/best_params.txt", "w", encoding="utf-8") as f:
            f.write(f"Best Trial Value: {study.best_value}\n\n")
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")

        print(f"Optuna visualization plots saved to {self.output_dir}")

    def _create_trials_dataframe(self, study):
        """
        Convert all trials into a pandas DataFrame for easier analysis.

        Args:
            study: Completed Optuna study object

        Returns:
            DataFrame containing all trials data
        """
        # Extract data from all trials
        trials_data = []

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            trial_data = {"trial_number": trial.number, "value": trial.value}
            trial_data.update(trial.params)
            trials_data.append(trial_data)

        # Convert to DataFrame
        df = pd.DataFrame(trials_data)

        return df

    def _plot_hyperparameter_scatter_plots(self, trials_df):
        """
        Create scatter plots for each hyperparameter vs objective value.

        Args:
            trials_df: DataFrame containing all trials data
        """
        # Get all hyperparameters (exclude trial_number and value)
        hyperparameters = [
            col for col in trials_df.columns if col not in ["trial_number", "value"]
        ]

        # Create a directory for these plots
        scatter_dir = f"{self.output_dir}/hyperparameter_scatter_plots"
        os.makedirs(scatter_dir, exist_ok=True)

        # Create scatter plot for each hyperparameter
        for param in hyperparameters:
            plt.figure(figsize=(10, 6))
            plt.scatter(trials_df[param], trials_df["value"], alpha=0.6)

            # Add trendline
            try:
                z = np.polyfit(trials_df[param], trials_df["value"], 1)
                p = np.poly1d(z)
                plt.plot(
                    sorted(trials_df[param]),
                    p(sorted(trials_df[param])),
                    "r--",
                    alpha=0.8,
                )
            except:
                pass  # Skip trendline if it can't be calculated

            plt.title(f"Effect of {param} on Objective Value")
            plt.xlabel(param)
            plt.ylabel("Objective Value (MSE)")
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.savefig(f"{scatter_dir}/{param}_scatter.png", dpi=300)
            plt.close()

        # Create a summary plot with top 4 most correlated parameters
        correlations = []
        for param in hyperparameters:
            corr = abs(np.corrcoef(trials_df[param], trials_df["value"])[0, 1])
            if not np.isnan(corr):
                correlations.append((param, corr))

        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Plot top 4 (or fewer if less than 4 parameters)
        top_params = [x[0] for x in correlations[: min(4, len(correlations))]]

        if len(top_params) > 0:
            _, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            for i, param in enumerate(top_params):
                if i >= 4:
                    break

                axes[i].scatter(trials_df[param], trials_df["value"], alpha=0.6)

                # Add trendline
                try:
                    z = np.polyfit(trials_df[param], trials_df["value"], 1)
                    p = np.poly1d(z)
                    axes[i].plot(
                        sorted(trials_df[param]),
                        p(sorted(trials_df[param])),
                        "r--",
                        alpha=0.8,
                    )
                except:
                    pass

                axes[i].set_title(f"Effect of {param}")
                axes[i].set_xlabel(param)
                axes[i].set_ylabel("Objective Value")
                axes[i].grid(True, linestyle="--", alpha=0.7)

            # Hide unused subplots
            for j in range(len(top_params), 4):
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(f"{scatter_dir}/top_parameters_summary.png", dpi=300)
            plt.close()

    def _plot_hyperparameter_distributions(self, trials_df):
        """
        Create distribution plots for each hyperparameter.

        Args:
            trials_df: DataFrame containing all trials data
        """
        # Get all hyperparameters (exclude trial_number and value)
        hyperparameters = [
            col for col in trials_df.columns if col not in ["trial_number", "value"]
        ]

        # Create a directory for these plots
        dist_dir = f"{self.output_dir}/hyperparameter_distributions"
        os.makedirs(dist_dir, exist_ok=True)

        # Create histogram for each hyperparameter
        for param in hyperparameters:
            plt.figure(figsize=(10, 6))

            # Create histogram
            plt.hist(trials_df[param], bins=20, alpha=0.7)

            # Mark the best value
            best_value = trials_df.loc[trials_df["value"].idxmin(), param]
            plt.axvline(
                x=best_value,
                color="r",
                linestyle="--",
                label=f"Best Value: {best_value:.6g}",
            )

            plt.title(f"Distribution of {param} Values")
            plt.xlabel(param)
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            plt.savefig(f"{dist_dir}/{param}_distribution.png", dpi=300)
            plt.close()

    def _plot_hyperparameter_correlations(self, trials_df):
        """
        Create correlation heatmap between hyperparameters.

        Args:
            trials_df: DataFrame containing all trials data
        """
        # Get all hyperparameters (exclude trial_number)
        columns = [col for col in trials_df.columns if col != "trial_number"]

        # Calculate correlation matrix
        corr_matrix = trials_df[columns].corr()

        # Create heatmap
        plt.figure(figsize=(12, 10))

        # Use a mask to hide the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Generate a custom diverging colormap
        cmap = plt.cm.coolwarm  # pylint: disable=no-member

        # Draw the heatmap

        try:
            _ = sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1,
                vmin=-1,
                center=0,
                square=True,
                linewidths=0.5,
                annot=True,
                fmt=".2f",
            )
        except ImportError:
            # Fallback to matplotlib if seaborn is not available
            plt.imshow(corr_matrix, cmap=cmap, vmax=1, vmin=-1)
            plt.colorbar()
            plt.xticks(range(len(columns)), columns, rotation=45)
            plt.yticks(range(len(columns)), columns)

        plt.title("Hyperparameter Correlation Matrix")
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/hyperparameter_correlation.png", dpi=300)
        plt.close()

    def _plot_pairwise_relationships(self, trials_df, study):
        """
        Create pairwise scatter plots for most important parameters.

        Args:
            trials_df: DataFrame containing all trials data
            study: Completed Optuna study object
        """
        try:
            # Try to get parameter importances
            importances = optuna.importance.get_param_importances(study)

            # Get top 3-4 most important parameters
            top_params = list(importances.keys())[: min(4, len(importances))]

            if len(top_params) < 2:
                return  # Need at least 2 parameters for pairwise plots

            # Create pairwise plots
            fig, axes = plt.subplots(len(top_params), len(top_params), figsize=(12, 12))

            # Normalize objective values for color mapping
            min_val = trials_df["value"].min()
            max_val = trials_df["value"].max()
            norm = plt.Normalize(min_val, max_val)

            for i, param1 in enumerate(top_params):
                for j, param2 in enumerate(top_params):
                    if i == j:  # Diagonal: show histogram
                        axes[i, j].hist(trials_df[param1], bins=15, alpha=0.7)
                        axes[i, j].set_title(param1)
                    else:  # Off-diagonal: show scatter plot
                        scatter = axes[i, j].scatter(
                            trials_df[param2],
                            trials_df[param1],
                            c=trials_df["value"],
                            cmap="viridis",
                            alpha=0.7,
                            norm=norm,
                        )

                        if i == len(top_params) - 1:  # Bottom row
                            axes[i, j].set_xlabel(param2)
                        if j == 0:  # Leftmost column
                            axes[i, j].set_ylabel(param1)

            # Add colorbar
            # pylint: disable=E0606
            cbar = fig.colorbar(scatter, ax=axes, orientation="vertical", pad=0.01)
            cbar.set_label("Objective Value")

            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/pairwise_relationships.png", dpi=300)
            plt.close()
        except:
            print("Could not generate pairwise relationship plots")

    def _create_interactive_visualization(self, study):
        """
        Create interactive HTML visualization if plotly is available.

        Args:
            study: Completed Optuna study object
        """
        try:
            # Create contour plots
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(f"{self.output_dir}/contour_plot.html")

            # Create slice plots
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(f"{self.output_dir}/slice_plot.html")

            # Create parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(f"{self.output_dir}/parallel_coordinate.html")

            # Create parameter importance plot
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(f"{self.output_dir}/param_importances.html")

            print(f"Interactive visualizations saved to {self.output_dir}")
        except ImportError:
            print("Plotly is not installed. Skipping interactive visualizations.")

    def save_model_performance(self, y_true, y_pred, set_name="test"):
        """
        Plot and save model performance metrics.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            set_name: Name of the dataset (test, validation, etc.)
        """
        # Convert to numpy arrays if they're tensors
        if hasattr(y_true, "cpu") and hasattr(y_pred, "cpu"):
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

        # Scatter plot of predicted vs actual values
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.title(f"Predicted vs Actual Values ({set_name} set)")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.axis("equal")
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/{set_name}_predictions.png", dpi=300)
        plt.close()

        # Calculate and save residuals plot
        residuals = y_pred - y_true

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")

        plt.title(f"Residuals Plot ({set_name} set)")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/{set_name}_residuals.png", dpi=300)
        plt.close()

        print(f"Model performance plots for {set_name} set saved to {self.output_dir}")
