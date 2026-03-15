
import numpy as np

np.random.seed(1313)

# Metrics ranges
temp = np.random.uniform(15, 45, (50, 24))
humidity = np.random.uniform(20, 95, (50, 24))
battery = np.random.uniform(10, 100, (50, 24))

# Combine into dataset (50 sensors, 24 hours, 3 metrics)
data = np.stack([temp, humidity, battery], axis=2)

# 2. Alert sensors
temp_alert = np.any(data[:, :, 0] > 40, axis=1)
humidity_alert = np.any(data[:, :, 1] > 90, axis=1)
alert_sensors = np.where(temp_alert | humidity_alert)[0]

# 3. Daily averages
daily_avg = data.mean(axis=1)

# 4. Hottest hour
avg_temp_per_hour = data[:, :, 0].mean(axis=0)
hottest_hour = np.argmax(avg_temp_per_hour)

# 5. Battery drain
battery_drop = data[:, 0, 2] - data[:, -1, 2]
critical_battery = np.where(battery_drop > 50)[0]

# 6. Normalize metrics per metric
mins = data.min(axis=(0,1), keepdims=True)
maxs = data.max(axis=(0,1), keepdims=True)
normalized = (data - mins) / (maxs - mins)

# 7. Save summary
np.savetxt("sensor_summary.csv", daily_avg, delimiter=",",
           header="temperature_avg,humidity_avg,battery_avg",
           comments="")

print("Alert sensors:", alert_sensors)
print("Hottest hour:", hottest_hour)
print("Sensors with critical battery drain:", critical_battery)
