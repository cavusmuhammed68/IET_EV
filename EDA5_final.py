import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\IET_code\station_data_dataverse_Final.csv'
data = pd.read_csv(file_path)

# Define colors for each category
colors = {
    'weekday': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf'],
    'platform': ['#ff7f0e', '#2ca02c'],
    'habitualUser': ['#d62728', '#9467bd'],
    'earlyAdopter': ['#8c564b', '#17becf']
}

# Calculate the percentage for each category in categorical columns
categorical_columns = ['weekday', 'platform', 'habitualUser', 'earlyAdopter']
category_percentages = {col: data[col].value_counts(normalize=True) * 100 for col in categorical_columns}

# Create the figure with the updated requirements
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each category
y_start = 0
y_labels = []
for category, values in category_percentages.items():
    y_end = y_start + len(values)
    bars = ax.barh(range(y_start, y_end), values.values, color=colors[category], edgecolor='none')
    for bar in bars:
        bar.set_edgecolor('none')
    for i, (label, value) in enumerate(values.items()):
        ax.text(value + 0.5, y_start + i, f'{value:.1f}%', va='center', fontsize=15)
        y_labels.append(f'{category}: {label}')
    y_start = y_end

# Set category labels
ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=15)
ax.invert_yaxis()

# Add grid, titles, and labels
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xlabel('Percentage (%)', fontsize=15)
#ax.set_title('EV Charging User Activity and Platform Preferences', fontsize=15)

# Remove the right border but keep the left border
ax.spines['right'].set_visible(False)

# Save the figure with high resolution
plt.savefig('user_behavior_preferences.png', dpi=1200, bbox_inches='tight')

plt.show()
