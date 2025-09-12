import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("weight-height.csv", delimiter=",", skip_header=1)
length_in = data[:, 0]
weight_lb = data[:, 1]

length_cm = length_in * 2.54
weight_kg = weight_lb * 0.453592

mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print("Mean length (cm):", mean_length)
print("Mean weight (kg):", mean_weight)

plt.hist(length_cm, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Student Lengths (cm)")
plt.xlabel("Length (cm)")
plt.ylabel("Frequency")
plt.show()
