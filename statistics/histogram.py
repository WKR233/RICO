import numpy as np
import matplotlib.pyplot as plt

#path = input()
most_common_dict = {}
path = "./data/scannet/scan1/"
for i in range(465):
    figpath = path + str(i).zfill(6) + '_depth.npy'
    file = np.load(figpath, encoding="latin1") # ndarray
    flattened_file = file.flatten().tolist()
    frequency, bins, _ = plt.hist(flattened_file, density=True, bins=100)
    max_frequency = frequency.argmax()
    most_common_bin = (bins[max_frequency]+bins[max_frequency+1])/2
    most_common_dict[i]=most_common_bin
    plt.savefig(path+str(i).zfill(6)+'_depth_histogram.png')
    plt.clf()

list = []
for i in range(465):
    list.append(most_common_dict[i])
plt.hist(list, density=True, bins=20)
plt.savefig(path+'total_depth_histogram.png')