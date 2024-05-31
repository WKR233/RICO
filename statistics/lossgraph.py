import numpy as np
import matplotlib.pyplot as plt

rgb_loss = []
depth_loss = []
normal_loss = []

with open('losses.txt', 'r') as file:
    for line in file:
        rgb_loss_index = line.find('rgb_loss')
        next_comma = line.find(',', rgb_loss_index)
        now_rgb_loss = float(line[rgb_loss_index+11:next_comma])
        rgb_loss.append(now_rgb_loss)

        depth_loss_index = line.find('depth_loss')
        next_comma = line.find(',', depth_loss_index)
        now_depth_loss = float(line[depth_loss_index+13:next_comma])
        depth_loss.append(now_depth_loss)

        normal_loss_index = line.find('normal_l1')
        next_comma = line.find(',', normal_loss_index)
        now_normal_loss = float(line[normal_loss_index+12:next_comma])
        normal_loss.append(now_normal_loss)

plt.hist(rgb_loss, density=True, bins=100)
plt.savefig('loss/rgb_loss.png')
plt.clf()
largest_rgb_loss_index = rgb_loss.index(max(rgb_loss))
print(largest_rgb_loss_index) #444

plt.hist(depth_loss, density=True, bins=100)
plt.savefig('loss/depth_loss.png')
plt.clf()
largest_depth_loss_index = depth_loss.index(max(depth_loss))
print(largest_depth_loss_index) #109

plt.hist(normal_loss, density=True, bins=100)
plt.savefig('loss/normal_loss.png')
plt.clf()
largest_normal_loss_index = normal_loss.index(max(normal_loss))
print(largest_normal_loss_index) #48