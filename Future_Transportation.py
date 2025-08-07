import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Set the style for seaborn
sns.set(style="whitegrid")

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Charging Session Costs')
plt.show()

# Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Distribution Plot of Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.histplot(y_test, color='blue', label='Actual', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color='red', label='Predicted', kde=True, stat="density", linewidth=0)
plt.legend()
plt.title('Distribution of Actual vs Predicted Charging Session Costs')
plt.show()

# Pair Plot of Features and Target
pairplot_data = data[features + [target]]
sns.pairplot(pairplot_data, diag_kind='kde')
plt.suptitle('Pair Plot of Features and Charging Session Costs', y=1.02)
plt.show()


