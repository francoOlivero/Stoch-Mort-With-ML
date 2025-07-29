import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r'C:\Users\frank\Downloads\PyUtilities\Stoch-Mort-With-ML\Docs\MLdf.xlsx'
mY_ML_Df= pd.read_excel(path)

# Set color palette from blue (-1) to red (1)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Define the delta columns and gender categories
deltas = ["delta_mx_Y_DT", "delta_mx_Y_RF", "delta_mx_Y_GB"]
genders = ["Male", "Female"]

# Create a 3 (rows = delta types) x 2 (columns = genders) grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)

# Loop through delta types and genders
for i, delta in enumerate(deltas):       # Rows
    for j, gender in enumerate(genders): # Columns
        ax = axes[i, j]

        # Filter data for current gender
        filtered = mY_ML_Df[mY_ML_Df["Gender"] == gender]

        # Pivot: rows = years, columns = age, values = delta
        pivot_data = filtered.pivot(index="Year", columns="Age", values=delta)

        # Plot heatmap
        sns.heatmap(
            pivot_data,
            ax=ax,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            cbar=(i == 0 and j == 1),  # Show only one colorbar (top-right)
        )

        # Title and labels
        ax.set_title(f"{delta} - {gender}")
        if j == 0:
            ax.set_ylabel("Year")
        else:
            ax.set_ylabel("")
        if i == 2:
            ax.set_xlabel("Age")
        else:
            ax.set_xlabel("")

plt.tight_layout()
plt.show()

print(mY_ML_Df)
# Define the delta columns and gender categories