import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\IET_code\station_data_dataverse_Final.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna()

# Encode categorical variables
data['weekday'] = data['weekday'].astype('category').cat.codes
data['platform'] = data['platform'].astype('category').cat.codes
data['facilityType'] = data['facilityType'].astype('category').cat.codes

# Define features and target
features = ['kwhTotal', 'startTime', 'endTime', 'chargeTimeHrs', 'weekday', 'platform', 'distance', 'managerVehicle', 'facilityType']
target = 'dollars'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = lr_model.score(X_test, y_test)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = rf_model.score(X_test, y_test)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = gb_model.score(X_test, y_test)

print(f'Linear Regression - MSE: {mse_lr}, R²: {r2_lr}')
print(f'Random Forest - MSE: {mse_rf}, R²: {r2_rf}')
print(f'Gradient Boosting - MSE: {mse_gb}, R²: {r2_gb}')

# Set the style for seaborn
sns.set(style="whitegrid")

# Function to clean data for residual plots
def clean_data(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

# Plot Actual vs Predicted values for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression\nActual vs Predicted')
plt.savefig('lr_actual_vs_predicted.png', dpi=1200)

# Random Forest
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest\nActual vs Predicted')
plt.savefig('rf_actual_vs_predicted.png', dpi=1200)

# Gradient Boosting
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gradient Boosting\nActual vs Predicted')
plt.savefig('gb_actual_vs_predicted.png', dpi=1200)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=1200)
plt.show()

# Plot Residuals for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
residuals_lr = y_test - y_pred_lr
y_pred_lr_clean, residuals_lr_clean = clean_data(y_pred_lr, residuals_lr)
sns.residplot(x=y_pred_lr_clean, y=residuals_lr_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Linear Regression\nResiduals Plot')
plt.savefig('lr_residuals.png', dpi=1200)

# Random Forest
plt.subplot(1, 3, 2)
residuals_rf = y_test - y_pred_rf
y_pred_rf_clean, residuals_rf_clean = clean_data(y_pred_rf, residuals_rf)
sns.residplot(x=y_pred_rf_clean, y=residuals_rf_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Random Forest\nResiduals Plot')
plt.savefig('rf_residuals.png', dpi=1200)

# Gradient Boosting
plt.subplot(1, 3, 3)
residuals_gb = y_test - y_pred_gb
y_pred_gb_clean, residuals_gb_clean = clean_data(y_pred_gb, residuals_gb)
sns.residplot(x=y_pred_gb_clean, y=residuals_gb_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Gradient Boosting\nResiduals Plot')
plt.savefig('gb_residuals.png', dpi=1200)

plt.tight_layout()
plt.savefig('residuals.png', dpi=1200)
plt.show()

# Plot Distribution of Actual vs Predicted values for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred_lr, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Linear Regression\nDistribution of Actual vs Predicted')
plt.savefig('lr_distribution.png', dpi=1200)

# Random Forest
plt.subplot(1, 3, 2)
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred_rf, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Random Forest\nDistribution of Actual vs Predicted')
plt.savefig('rf_distribution.png', dpi=1200)

# Gradient Boosting
plt.subplot(1, 3, 3)
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred_gb, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Gradient Boosting\nDistribution of Actual vs Predicted')
plt.savefig('gb_distribution.png', dpi=1200)

plt.tight_layout()
plt.savefig('distribution.png', dpi=1200)
plt.show()

# Analyze charging patterns by time of day
plt.figure(figsize=(12, 6))
sns.histplot(data['startTime'], bins=24, kde=True, color='skyblue')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Charging Sessions')
plt.title('Charging Patterns by Time of Day')
plt.xticks(range(0, 24))
plt.savefig('charging_patterns_by_time_of_day.png', dpi=1200)
plt.show()

# Analyze charging patterns by day of the week
plt.figure(figsize=(12, 6))
sns.countplot(x='weekday', data=data, order=sorted(data['weekday'].unique()), palette='viridis')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Charging Sessions')
plt.title('Charging Patterns by Day of the Week')
plt.savefig('charging_patterns_by_day_of_week.png', dpi=1200)
plt.show()

# Analyze the distribution of distances traveled to the charging station
plt.figure(figsize=(12, 6))
sns.histplot(data['distance'], bins=30, kde=True, color='lightgreen')
plt.xlabel('Distance (miles)')
plt.ylabel('Number of Charging Sessions')
plt.title('Distribution of Distances Traveled to Charging Station')
plt.savefig('distribution_of_distances_traveled.png', dpi=1200)
plt.show()

# Relationship between distance traveled and cost
plt.figure(figsize=(12, 6))
sns.scatterplot(x='distance', y='dollars', data=data, alpha=0.5, color='coral')
plt.xlabel('Distance (miles)')
plt.ylabel('Charging Cost (dollars)')
plt.title('Relationship Between Distance Traveled and Charging Cost')
plt.savefig('relationship_between_distance_and_cost.png', dpi=1200)
plt.show()

# Analyze the distribution of charging session durations
plt.figure(figsize=(12, 6))
sns.histplot(data['chargeTimeHrs'], bins=30, kde=True, color='gold')
plt.xlabel('Charging Duration (hours)')
plt.ylabel('Number of Charging Sessions')
plt.title('Distribution of Charging Session Durations')
plt.savefig('distribution_of_charging_session_durations.png', dpi=1200)
plt.show()

# If needed, save all figures again with high resolution explicitly set in each savefig call
# For example:
# plt.savefig('figure_name.png', dpi=1200)


