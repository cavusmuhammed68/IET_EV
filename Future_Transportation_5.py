import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report
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

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Define cost categories based on percentiles
cost_thresholds = np.percentile(y_test, [33, 66])
def categorize_cost(cost):
    if cost < cost_thresholds[0]:
        return 'Low'
    elif cost < cost_thresholds[1]:
        return 'Medium'
    else:
        return 'High'

# Apply categorization to actual and predicted values
y_test_cat = y_test.apply(categorize_cost)
y_pred_lr_cat = pd.Series(y_pred_lr).apply(categorize_cost)
y_pred_rf_cat = pd.Series(y_pred_rf).apply(categorize_cost)
y_pred_gb_cat = pd.Series(y_pred_gb).apply(categorize_cost)

# Generate confusion matrices
cm_lr = confusion_matrix(y_test_cat, y_pred_lr_cat, labels=['Low', 'Medium', 'High'])
cm_rf = confusion_matrix(y_test_cat, y_pred_rf_cat, labels=['Low', 'Medium', 'High'])
cm_gb = confusion_matrix(y_test_cat, y_pred_gb_cat, labels=['Low', 'Medium', 'High'])

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0], cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
axes[0].set_title('Linear Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[1], cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

sns.heatmap(cm_gb, annot=True, fmt='d', ax=axes[2], cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
axes[2].set_title('Gradient Boosting')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Print classification reports
print("Linear Regression Classification Report")
print(classification_report(y_test_cat, y_pred_lr_cat, labels=['Low', 'Medium', 'High'], target_names=['Low', 'Medium', 'High']))

print("Random Forest Classification Report")
print(classification_report(y_test_cat, y_pred_rf_cat, labels=['Low', 'Medium', 'High'], target_names=['Low', 'Medium', 'High']))

print("Gradient Boosting Classification Report")
print(classification_report(y_test_cat, y_pred_gb_cat, labels=['Low', 'Medium', 'High'], target_names=['Low', 'Medium', 'High']))
