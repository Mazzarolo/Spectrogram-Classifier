import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "results/metrics_results.csv"
df = pd.read_csv(file_path)

df.columns = ["accuracy", "precision", "recall", "f1_score"]

df_melted = df.melt(var_name="Metric", value_name="Value")

plt.figure(figsize=(8, 6))
sns.boxplot(x="Metric", y="Value", data=df_melted, palette="Set2")

sns.stripplot(x="Metric", y="Value", data=df_melted, color="black", alpha=0.6)

plt.title("Metric Distribution")
plt.xlabel("Metric")
plt.ylabel("Value")
plt.grid(True)

plt.show()