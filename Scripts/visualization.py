import matplotlib.pyplot as plt
from model_regression import model_list
import json
import os
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

model_list=model_list
def visualize_model_performance(model_list):
    # Dictionary to store train and test scores for each model
    train_scores = []
    test_scores = []
    model_names = []

    # Load metrics for each model from saved files
    for model in model_list:
        model_name = str(model)[:-2]
        model_names.append(model_name)

        # Load the metrics
        metrics_file = f"results/{model_name}/metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Append the scores to the lists
        train_scores.append(metrics["train_score"])
        test_scores.append(metrics["test_score"])

    # Plotting the performance
    index = range(len(model_list))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))

    # Plotting train and test scores
    plt.bar(index, train_scores, bar_width, label='Train R2 Score', color='b')
    plt.bar([i + bar_width for i in index], test_scores, bar_width, label='Test R2 Score', color='g')

    # Adding labels and titles
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('Comparison of R2 Scores Across Different Models')
    plt.xticks([i + bar_width / 2 for i in index], model_names, rotation=45)
    plt.legend()

   
    plt.tight_layout()

    folder ='results/figures'
    filename = 'Comparison_model.png'

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    plt.savefig(fname=file_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_heatmap_for_model_performance():
    # Collect data for the heatmap
    data = []

    model_list=[
    
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    GradientBoostingRegressor()

]
    
    for model in model_list:
        model_name = str(model)[:-2]

        # Load the metrics
        metrics_file = f"results/{model_name}/metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Load the hyperparameters
        params_file = f"results/{model_name}/hyperparameters.json"
        with open(params_file) as f:
            params = json.load(f)

        # Append data to the list
        data.append({
            'Model': model_name,
            'Train R2 Score': metrics["train_score"],
            'Test R2 Score': metrics["test_score"],
             **params 
        })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    if 'max_depth' in df.columns:
        pivot_table = df.pivot(index="Model", columns="max_depth", values="Test R2 Score")

        # Plotting the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", linewidths=0.5)
        plt.title("Heatmap of Test R2 Score across Models and max_depth Hyperparameter", fontsize=16)
        plt.xlabel("max_depth")
        plt.ylabel("Model")
        plt.tight_layout()

        # Save the figure
        folder = 'results/figures'
        filename = 'Heatmap_Test_R2_Score.png'
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)

        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("The selected hyperparameter 'max_depth' is not available for all models.")
 

if __name__=="__main__":
    visualize_model_performance(model_list)
    create_heatmap_for_model_performance()