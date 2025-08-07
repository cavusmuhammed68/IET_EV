import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load the dataset
file_path = r'C:\Users\nfpm5\OneDrive - Northumbria University - Production Azure AD\Desktop\IET_code\station_data_dataverse_Final2.csv'
data = pd.read_csv(file_path)

# Rename columns for clarity
data = data.rename(columns={
    'managerVehicle': 'Manager Vehicle',
    'facilityType': 'Facility Type',
    'userId': 'User ID',
    'habitualUser': 'Habitual User',
    'totalsessions': 'Total Sessions',
    'earlyAdopter': 'Early Adopter',
    'reportedZip': 'Reported Zip'
})

# Convert 'startTime' to integer (assuming 'startTime' contains hour values as integers)
data['startTime'] = data['startTime'].astype(int)

# Set the style for seaborn
sns.set(style="whitegrid")

# Order of the days starting with Monday
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot diurnal profile for each day of the week
plt.figure(figsize=(18, 24))
for i, day in enumerate(ordered_days):
    plt.subplot(4, 2, i + 1)
    day_data = data[data['weekday'] == day]['startTime']
    sns.histplot(day_data, bins=24, kde=True, color='skyblue')
    plt.title(f'{day} (N={len(day_data)})', fontsize=20)
    plt.xlabel('Hour of Day', fontsize=20)
    plt.ylabel('Number of Charging Sessions', fontsize=20)
    plt.xticks([0, 5, 10, 15, 20, 23], fontsize=20)
    plt.yticks(fontsize=20)
    if day in ['Saturday', 'Sunday']:
        plt.ylim(0, data[data['weekday'] == 'Saturday']['startTime'].value_counts().max())
    else:
        plt.ylim(0, 100)
    plt.tight_layout()

plt.savefig('diurnal_profile_by_day.png', dpi=1200)
plt.show()

# ANOVA to test if there are significant differences between days
grouped_data = [data[data['weekday'] == day]['startTime'] for day in ordered_days]
anova_result = f_oneway(*grouped_data)

print(f'ANOVA result: F-value = {anova_result.statistic}, p-value = {anova_result.pvalue}')
