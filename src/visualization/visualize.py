import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
import os

def plot_and_save_tree(model, feature_names, output_path):
    """
    Plot a decision tree and save it to a file.
    """
    try:
        print(f"Generating decision tree plot...")
        # Create a figure for the decision tree plot
        plt.figure(figsize=(20, 10))
        # Plot the decision tree with specified options
        plot_tree(
            model,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            fontsize=10
        )

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the plot to the specified file path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free resources
        print(f"Decision tree plot saved to {output_path}")

    except Exception as e:
        # Handle any errors during plotting or saving
        print(f"Error generating or saving tree plot: {e}")