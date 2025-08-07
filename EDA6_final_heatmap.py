import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\IET_code\station_data_dataverse_Final.csv'
data = pd.read_csv(file_path)

# Rename columns
data = data.rename(columns={
    'managerVehicle': 'Manager Vehicle',
    'facilityType': 'Facility Type',
    'userId': 'User ID',
    'habitualUser': 'Habitual User',
    'totalsessions': 'Total Sessions',
    'earlyAdopter': 'Early Adopter',
    'reportedZip': 'Reported Zip'
})

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display summary statistics
print("\nSummary statistics of the dataset:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Select only numeric columns for correlation matrix
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = data[numeric_cols].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# Set the style for seaborn
sns.set(style="whitegrid")

# Pairplot to visualize relationships between features (reduced DPI)
sns.pairplot(data[numeric_cols])
plt.savefig(r'C:\Users\cavus\IdeaProjects\Future_Transportation\pairplot.png', dpi=300)  # Reduced DPI for pairplot to avoid large image size issue
plt.show()

# Heatmap of the correlation matrix (lower triangle only)
plt.figure(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": .8}, square=True, annot_kws={"size": 14})
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(r'C:\Users\cavus\IdeaProjects\Future_Transportation\heatmap_correlation_matrix.png', dpi=1200, bbox_inches='tight')
plt.show()

