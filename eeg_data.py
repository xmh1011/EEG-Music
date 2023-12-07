import pickle
import numpy as np


# Function to load data from each participant file
def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p


files = []
for n in range(1, 20):
    s = str(n)
    files.append(s)

labels = []
data = []

for i in files:
    filename = "./data/sample_" + i + ".dat"
    trial = read_eeg_signal_from_file(filename)
    labels.append(trial['labels'])
    data.append(trial['data'])

# Re-shape arrays into desired shapes
labels = np.array(labels)
print(labels.shape)

data = np.array(data)
print(data.shape)
