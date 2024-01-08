import numpy as np
import matplotlib.pyplot as plt

dep = [9,9,9,8,8,8,7,7,7,7,7,6,6,6,5,5,5,4,4,3,3,3,2,1,1,1,1,1,0,0,0]
arr = [4,0,0,9,4,4,9,4,4,4,4,2,2,0,3,3,0,2,2,4,4,1,5,8,8,7,4,3,7,4,2]
times = [248, 188, 22, 157, 117,7,109,0,202, 166, 41,83,223,94,92, 52, 79,18,64,125,131,289,163,13,300, 1, 14, 38,97,188, 249]

depval, depcount = np.unique(dep, return_counts=True)
arrval, arrcount = np.unique(arr, return_counts=True)

arrval = np.insert(arrval, 6, 6)
arrcount = np.insert(arrcount, 6, 0)

print(depval, arrval, arrcount)

ind = np.arange(10)

width = 0.3  
plt.bar(ind, depcount , width, label='Departure Counts')
plt.bar(ind + width, arrcount, width, label='Arrival Counts')

plt.xlabel('Departure/ Arrival Node')
plt.ylabel('# Departure/Arrival')

plt.xticks(ind + width / 2, ('Helipad', 'Airport','Rural A','City Hub',
                             'School','Sports Ctr','Rural B',
                             'IAD Conn', 'BWI Conn (N)','BWI Conn (S)'))
plt.xticks(rotation = 45)
plt.legend()

# plt.bar(depval, depcount)
# plt.bar(arrval, arrcount)


plt.show()

interval = np.arange(310, step = 10)
print(interval)

plt.hlines(1,1,300)  # Draw a horizontal line
plt.xlim(0,300)
plt.ylim(0.95,1.05)

ax = plt.gca()
ax.get_yaxis().set_visible(False)

y = np.ones(np.shape(times))   # Make all y values the same
y2 = np.ones(np.shape(interval))
plt.plot(interval,y2,'|',ms = 20, c='red', linewidth=5, label='10s Interval') 
plt.plot(times,y,'*', linewidth=5, label='Departure Intervals',c='green', markersize=10)

plt.legend()
plt.xlabel('Time (s)')

plt.show()