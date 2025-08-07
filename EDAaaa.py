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

# Plot diurnal profile for each day of the week
plt.figure(figsize=(18, 12))
for i, day in enumerate(sorted(data['weekday'].unique())):
    plt.subplot(4, 2, i + 1)
    sns.histplot(data[data['weekday'] == day]['startTime'], bins=24, kde=True, color='skyblue')
    plt.title(f'{day}', fontsize=20)
    plt.xlabel('Hour of Day', fontsize=15)
    plt.ylabel('Number of Charging Sessions', fontsize=15)
    plt.xticks([0, 5, 10, 15, 20, 23], fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

plt.savefig('diurnal_profile_by_day.png', dpi=600)
plt.show()

# ANOVA to test if there are significant differences between days
grouped_data = [data[data['weekday'] == day]['startTime'] for day in sorted(data['weekday'].unique())]
anova_result = f_oneway(*grouped_data)

print(f'ANOVA result: F-value = {anova_result.statistic}, p-value = {anova_result.pvalue}')
