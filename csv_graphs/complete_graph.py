import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths and corresponding labels
files = {
    "classification_loss": "csv_graphs/csv_files/classification_loss.csv",
    "learning_rate": "csv_graphs/csv_files/learning_rate.csv",
    "localization_loss": "csv_graphs/csv_files/localization_loss.csv",
    "regularization_loss": "csv_graphs/csv_files/regularization_loss.csv",
    "total_loss": "csv_graphs/csv_files/total_loss.csv"
}

plt.figure(figsize=(12, 8))

# Loop through each file, load the data, and plot the graph
for label, path in files.items():
    df = pd.read_csv(path)
    # Optionally, print column names and head for debugging:
    print(f"Columns in {label}:", df.columns)
    print(df.head())
    
    # Plot each line; update "Step" and "Value" to match your actual column names if different.
    sns.lineplot(data=df, x="Step", y="Value", label=label)

plt.title("Training Metrics")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend(title="Metrics")
plt.show()
