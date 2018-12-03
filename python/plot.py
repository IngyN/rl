# this code is to plot the summary from training
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os import walk

#read files available
f = []
for (dirpath, dirnames, filenames) in walk('python/logs'):
    f.extend(filenames)
    break

print(f)

file = input('Please select file: ')

file = f[int(file)-1]

with open('python/logs/'+file, 'rb') as fp:
	summary = pickle.load(fp)

summary = summary[1:-1]
summary = summary.replace(',', '')
summary = summary.split(' ')
summary = [int(e) for e in summary]


# Fixing random state for reproducibility
np.random.seed(22)

# the histogram of the data
plt.plot(np.arange(0, len(summary[:91])),summary[:91])


plt.xlabel('Episode')
plt.ylabel('Number of steps')
plt.title('Episodes and steps')
plt.grid(False)
plt.show()