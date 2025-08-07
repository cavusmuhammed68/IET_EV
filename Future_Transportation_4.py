import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to clean data by removing NaN and infinite values
def clean_data(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

# Clean data for residual plots
residuals_lr = y_test - y_pred_lr
residuals_rf = y_test - y_pred_rf
residuals_gb = y_test - y_pred_gb

y_pred_lr_clean, residuals_lr_clean = clean_data(y_pred_lr, residuals_lr)
y_pred_rf_clean, residuals_rf_clean = clean_data(y_pred_rf, residuals_rf)
y_pred_gb_clean, residuals_gb_clean = clean_data(y_pred_gb, residuals_gb)

# Set the style for seaborn
sns.set(style="whitegrid")

# Plot Actual vs Predicted values for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')
plt.savefig('linear_regression_actual_vs_predicted.png', dpi=1200)

# Random Forest
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')
plt.savefig('random_forest_actual_vs_predicted.png', dpi=1200)

# Gradient Boosting
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_gb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Gradient Boosting')
plt.savefig('gradient_boosting_actual_vs_predicted.png', dpi=1200)

plt.tight_layout()
plt.show()

# Plot Residuals for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 3, 1)
sns.residplot(x=y_pred_lr_clean, y=residuals_lr_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Linear Regression')
plt.savefig('linear_regression_residuals.png', dpi=1200)

# Random Forest
plt.subplot(1, 3, 2)
sns.residplot(x=y_pred_rf_clean, y=residuals_rf_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Random Forest')
plt.savefig('random_forest_residuals.png', dpi=1200)

# Gradient Boosting
plt.subplot(1, 3, 3)
sns.residplot(x=y_pred_gb_clean, y=residuals_gb_clean, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Gradient Boosting')
plt.savefig('gradient_boosting_residuals.png', dpi=1200)

plt.tight_layout()
plt.show()

# Plot Distribution of Actual vs Predicted values for each model
plt.figure(figsize=(18, 6))

# Linear Regression
plt.subplot(1, 2, 1)
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred_lr, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Linear Regression')
plt.savefig('linear_regression_distribution.png', dpi=1200)

# Random Forest
plt.subplot(1, 2, 2)
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred_rf, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Random Forest')
plt.savefig('random_forest_distribution.png', dpi=1200)

# Gradient Boosting
#plt.subplot(1, 3, 3)
#sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
#sns.histplot(y_pred_gb, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
#plt.legend()
#plt.title('Gradient Boosting')
#plt.savefig('gradient_boosting_distribution.png', dpi=1200)

plt.tight_layout()
plt.show()
