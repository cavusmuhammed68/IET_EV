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
sns.histplot(data['distance'], bins=30, kde=True, color='green')
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
sns.histplot(data['chargeTimeHrs'], bins=30, kde=True, color='blue')
plt.xlabel('Charging Duration (hours)')
plt.ylabel('Number of Charging Sessions')
plt.title('Distribution of Charging Session Durations')
plt.savefig('distribution_of_charging_session_durations.png', dpi=1200)
plt.show()
