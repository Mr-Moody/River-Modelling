import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV files
df_actual = pd.read_csv('csv_files/Jurua_20170710.csv')
df_predicted = pd.read_csv('csv_files/predicted_Jurua_20170710.csv')
df_initial = pd.read_csv('csv_files/Jurua_19871012.csv')

# Extract relevant columns, 
actual_data = df_actual['centerline_x, centerline_y'].to_numpy()
predicted_data = df_predicted['centerline_x, centerline_y'].to_numpy()
initial_data = df_initial['centerline_x, centerline_y'].to_numpy()

# Plot predicted vs actual
plt.figure()
plt.plot(initial_data, label='Initial', color='green', linestyle=':')
plt.plot(actual_data, label='Actual', color='blue')
plt.plot(predicted_data, label='Predicted', color='orange', linestyle='--')
plt.xlabel('Easting(m)')
plt.ylabel('Northing(m)')
plt.title('Predicted vs Actual Centerline X, Y')
plt.legend()
plt.grid()
plt.show()

# Plotting each separately
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

ax1.plot(initial_data, label='Initial', color='green', linestyle='--')
ax1.plot(actual_data, label='Actual', color='blue')
ax1.set_xlabel('Easting(m)')
ax1.set_ylabel('Northing(m)')
ax1.set_title('Actual Centerline X, Y')
ax1.legend()
ax1.grid()  

ax2.plot(initial_data, label='Initial', color='green', linestyle='--')
ax2.plot(predicted_data, label='Predicted', color='orange')
ax2.set_xlabel('Easting(m)')
ax2.set_ylabel('Northing(m)')
ax2.set_title('Predicted Centerline X, Y')
ax2.legend()
ax2.grid()  

plt.tight_layout()
plt.show()

# Calculate and plot root mean squared error for centerline positions over time
rmse = np.sqrt(np.mean((predicted_data - actual_data) ** 2, axis=1))
plt.figure()
plt.plot(rmse, label='Squared Error', color='red')
plt.xlabel('Time Step')
plt.ylabel('Squared Error')
plt.title('Mean Squared Error of Centerline Positions Over Time')
plt.legend()
plt.grid()
plt.show()

