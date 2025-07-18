import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn style for clean and aesthetic plots
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

# Path to the main dataset folder
database_path = "database"

# Initialize a dictionary to count occurrences for each class and ripeness level
count_dict = {
    "ClassA": [0, 0, 0, 0],
    "ClassB": [0, 0, 0, 0],
    "ClassC": [0, 0, 0, 0],
    "ClassD": [0, 0, 0, 0],
}

# Regex pattern to extract class and ripeness stage from file names
# Example: "123_ClassB_Ripe3_2100g.jpg" → ClassB and Ripe3
pattern = re.compile(r"Class([A-D])_Ripe([1-4])")

# Walk through the dataset directory
for root, dirs, files in os.walk(database_path):
    if "crop" in root.lower():  # Skip any folder that contains 'crop' in the path
        continue
    for file in files:
        match = pattern.search(file)
        if match:
            class_letter = "Class" + match.group(1)  # ClassA, ClassB, etc.
            ripe_index = int(match.group(2)) - 1  # Convert Ripe1 → index 0
            count_dict[class_letter][ripe_index] += 1

# Convert the count dictionary into a DataFrame
df = pd.DataFrame(count_dict, index=["Ripe1", "Ripe2", "Ripe3", "Ripe4"])
df.index.name = "Ripeness Stage"

# Melt the DataFrame into long format for Seaborn plotting
df_melted = df.reset_index().melt(
    id_vars="Ripeness Stage", var_name="Class", value_name="Count"
)

# Plot the barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_melted, x="Ripeness Stage", y="Count", hue="Class")

# Annotate each bar with its count
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

# Set title and axis labels
plt.title("Distribution of Fruit Classes by Ripeness Stage", fontsize=16, weight="bold")
plt.xlabel("Ripeness Stage")
plt.ylabel("Number of Samples")
plt.legend(title="Fruit Class", loc="upper right")
plt.tight_layout()
plt.show()
