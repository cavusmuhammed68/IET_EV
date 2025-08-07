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
plt.figure(figsize=(16, 10))
sns.pairplot(data[numeric_cols])
plt.savefig('pairplot.png', dpi=600)  # Reduced DPI for pairplot to avoid large image size issue
plt.show()

# Heatmap of the correlation matrix (lower triangle only)
plt.figure(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": .8}, square=True, annot_kws={"size": 14})
plt.savefig('heatmap_correlation_matrix.png', dpi=600, bbox_inches='tight')
plt.show()

# Histograms of all numerical features
data[numeric_cols].hist(bins=30, figsize=(20, 15), edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.savefig('histograms_numerical_features.png', dpi=600)
plt.show()

# Boxplots to check for outliers in numerical features
plt.figure(figsize=(16, 10))
sns.boxplot(data=data[numeric_cols], palette="Set3")
plt.savefig('boxplots_numerical_features.png', dpi=600)
plt.show()

# Analyzing the distribution of the target variable 'dollars'
plt.figure(figsize=(10, 6))
sns.histplot(data['dollars'], bins=30, kde=True, color='blue')
plt.xlabel('Charging Cost (dollars)')
plt.ylabel('Frequency')
plt.savefig('distribution_charging_costs.png', dpi=600)
plt.show()

# Analyzing the distribution of 'kwhTotal'
plt.figure(figsize=(10, 6))
sns.histplot(data['kwhTotal'], bins=30, kde=True, color='green')
plt.xlabel('Total Energy Consumption (kWh)')
plt.ylabel('Frequency')
plt.savefig('distribution_energy_consumption.png', dpi=600)
plt.show()

# Analyzing the distribution of 'chargeTimeHrs'
plt.figure(figsize=(10, 6))
sns.histplot(data['chargeTimeHrs'], bins=30, kde=True, color='red')
plt.xlabel('Charging Duration (hours)')
plt.ylabel('Frequency')
plt.savefig('distribution_charging_duration.png', dpi=600)
plt.show()

# Analyze the distribution of distances traveled to the charging station
plt.figure(figsize=(12, 6))
sns.histplot(data['distance'], bins=30, kde=True, color='lightgreen')
plt.xlabel('Distance (miles)')
plt.ylabel('Number of Charging Sessions')
plt.savefig('distribution_of_distances_traveled.png', dpi=600)
plt.show()

# Relationship between distance traveled and cost
plt.figure(figsize=(12, 6))
sns.scatterplot(x='distance', y='dollars', data=data, alpha=0.5, color='coral')
plt.xlabel('Distance (miles)')
plt.ylabel('Charging Cost (dollars)')
plt.savefig('relationship_between_distance_and_cost.png', dpi=600)
plt.show()

# Analyze charging patterns by time of day
plt.figure(figsize=(12, 6))
sns.histplot(data['startTime'], bins=24, kde=True, color='skyblue')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Charging Sessions')
plt.xticks(range(0, 24))
plt.savefig('charging_patterns_by_time_of_day.png', dpi=600)
plt.show()

# Analyze charging patterns by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(x='weekday', data=data, order=sorted(data['weekday'].unique()), palette='viridis')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Charging Sessions')
plt.savefig('charging_patterns_by_day_of_week.png', dpi=600)
plt.show()
