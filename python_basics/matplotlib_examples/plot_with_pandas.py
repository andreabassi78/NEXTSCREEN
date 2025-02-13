import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the Excel file (Make sure the file path is correct)

path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(path,"my_table.xlsx")

df = pd.read_excel(file_path)

print(df)

# Plot the data (assuming we have 'Date' and 'Sales' columns)
plt.figure(figsize=(10, 5))
plt.plot(df["time"], df["Intensity"], marker="o", linestyle="-", color="b", label="Intensity over time")

# Formatting the plot
plt.xlabel("time")
plt.ylabel("Intensity")
plt.title("Signal")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)

# Show the plot
plt.show()