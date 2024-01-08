import numpy as np
import matplotlib.pyplot as plt

data02_times = np.array([1.16,1.4,1.6,1.74,1.78,1.03,6.64,1.22,1.19,1.54,1.16])
data64_times = np.array([4.01,8.36,13.97,5.21,7.09,4.52,9.71,2.74,5,8.13,5.44])
data45_times = np.array([0.67,0.64,3.7,1,0.54,0.77,0.7,2.8,0.84,0.74,1.71])
data81_times = np.array([1.03,1.11,2.29,1.27,1.45,1.71,1.61,1.28,3.93,1.36,3.85])
data09_times = np.array([2.21,2.59,1.31,0.67,0.75,1.39,1.15,1.01,1.57,2.59,2.46])


data = [data02_times,data64_times,data45_times,data81_times,data09_times]

print(np.mean(data,axis=1))
fig = plt.figure(figsize=(6,2))
# ax = fig.add_axes([0,0,1,1])
plt.boxplot(data,labels=['Route 0-2','Route 6-4','Route 4-5','Route 8-1','Route 0-9'], notch=False,vert=0)

plt.grid(which='both',axis='x')

plt.xlabel('Time (s)')
plt.title('RRT* Runtime')
plt.show()

data02_distances = np.array([14.83,14.46,14.2,16.09,15.78,14.64,14.47,14.45,14.35,15.48,14.46])
data64_distances = np.array([22.24,25.83,23.46,23.69,26.14,23.68,23.99,22.89,23.5,26.41,22])
data45_distances = np.array([12.01,12.05,12.43,11.84,12.3,10.84,11.42,12.4,13.11,12.34,12.54])
data81_distances = np.array([11.85,11.57,11.58,12.61,16.98,13.6,14.95,11.54,18.49,13.93,13.93])
data09_distances = np.array([11.37,13.48,15.64,15.57,9.91,15.66,15.71,10.01,15,15.31,10.05])


data = [data02_distances,data64_distances,data45_distances,data81_distances,data09_distances]

print(np.mean(data,axis=1))
fig = plt.figure(figsize=(6,2))
# ax = fig.add_axes([0,0,1,1])
plt.boxplot(data,labels=['Route 0-2','Route 6-4','Route 4-5','Route 8-1','Route 0-9'], notch=False,vert=0)

plt.grid(which='both',axis='x')
plt.xlabel('Distance (km)')
plt.title('RRT* Distances')

plt.show()