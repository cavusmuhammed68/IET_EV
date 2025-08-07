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

# Set the style for seaborn
sns.set(style="whitegrid")

# Combined plots for distributions
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

# Distribution of Charging Duration
sns.histplot(data['chargeTimeHrs'], bins=30, kde=True, color='red', ax=axs[0])
axs[0].set_xlabel('Charging Duration (hours)', fontsize=25)
axs[0].set_ylabel('Frequency', fontsize=25)
axs[0].set_title('(a)', fontsize=25)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# Distribution of Total Energy Consumption
sns.histplot(data['kwhTotal'], bins=30, kde=True, color='green', ax=axs[1])
axs[1].set_xlabel('Total Energy Consumption (kWh)', fontsize=25)
axs[1].set_ylabel('Frequency', fontsize=25)
axs[1].set_title('(b)', fontsize=25)
axs[1].tick_params(axis='both', which='major', labelsize=25)

# Distribution of Charging Costs
sns.histplot(data['dollars'], bins=30, kde=True, color='blue', ax=axs[2])
axs[2].set_xlabel('Charging Cost (dollars)', fontsize=25)
axs[2].set_ylabel('Frequency', fontsize=25)
axs[2].set_title('(c)', fontsize=25)
axs[2].tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.savefig('combined_distributions.png', dpi=1200)
plt.show()

# Combined plots for relationship and sessions by day of the week
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Relationship between distance traveled and cost
sns.scatterplot(x='distance', y='dollars', data=data, alpha=0.5, color='coral', ax=axs[0])
axs[0].set_xlabel('Distance (miles)', fontsize=25)
axs[0].set_ylabel('Charging Cost (dollars)', fontsize=25)
axs[0].set_title('(a)', fontsize=20)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# Number of charging sessions by day of the week
sns.countplot(x='weekday', data=data, order=sorted(data['weekday'].unique()), palette='viridis', ax=axs[1])
axs[1].set_xlabel('Day of the Week', fontsize=25)
axs[1].set_ylabel('Number of Charging Sessions', fontsize=25)
axs[1].set_title('(b)', fontsize=25)
axs[1].tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.savefig('combined_relationship_sessions_by_day.png', dpi=1200)
plt.show()

# Combined plots for sessions by hour of day and distances
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Number of charging sessions by hour of day
sns.histplot(data['startTime'], bins=24, kde=True, color='skyblue', ax=axs[0])
axs[0].set_xlabel('Hour of Day', fontsize=25)
axs[0].set_ylabel('Number of Charging Sessions', fontsize=25)
axs[0].set_xticks([0, 5, 10, 15, 20, 23])
axs[0].set_title('(a)', fontsize=25)
axs[0].tick_params(axis='both', which='major', labelsize=25)

# Number of charging sessions by distance
sns.histplot(data['distance'], bins=30, kde=True, color='lightgreen', ax=axs[1])
axs[1].set_xlabel('Distance (miles)', fontsize=25)
axs[1].set_ylabel('Number of Charging Sessions', fontsize=25)
axs[1].set_title('(b)', fontsize=25)
axs[1].tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.savefig('combined_sessions_by_hour_distance.png', dpi=1200)
plt.show()
