import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Calculate total number and percentage for the plot
total_sessions = len(data)
total_distance = data['distance'].sum()

# Plot the relationship between distance traveled and cost
plt.figure(figsize=(18, 12))
sns.scatterplot(x='distance', y='dollars', data=data, alpha=0.5, color='darkblue', linewidth=2)
plt.xlabel('Distance (miles)', fontsize=25)
plt.ylabel('Charging Cost (dollars)', fontsize=25)
#plt.title('Relationship between Distance Traveled and Charging Cost', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
#plt.text(0.95, 0.95, f'Total Sessions: {total_sessions}\nTotal Distance: {total_distance:.2f} miles',
         #verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes, fontsize=20)

plt.tight_layout()
plt.savefig('relationship_between_distance_and_cost.png', dpi=1200)
plt.show()
