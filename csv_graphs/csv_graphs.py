import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("csv_graphs/csv_files/classification_loss.csv")

# Print column names to verify
print("Columns in CSV:", df.columns)

# Check first few rows
print(df.head())

# Update these column names with actual ones from your CSV
sns.lineplot(data=df, x="Step", y="Value")

plt.title("Classification Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()
