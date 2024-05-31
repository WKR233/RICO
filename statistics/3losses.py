from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

rendering_box, sementic_box, depth_box, normal_box = (0, 0, 388, 774), (388, 0, 776, 774), (776, 0, 1164, 774), (1164, 0, 1552, 774)
model_box, gt_box = (2, 2, 386, 386), (2, 388, 386, 772)

# split the images
merge_image = Image.open(sys.argv[1])
rendering_image = merge_image.crop(rendering_box)
rendering_image.save('rendering/'+'rendering_'+sys.argv[1])
model_rendering_image = rendering_image.crop(model_box)
model_rendering_image.save('rendering/'+'model_rendering_'+sys.argv[1])
rendering_image = Image.open('rendering/'+'rendering_'+sys.argv[1])
gt_rendering_image = rendering_image.crop(gt_box)
gt_rendering_image.save('rendering/'+'gt_rendering_'+sys.argv[1])

merge_image = Image.open(sys.argv[1])
sementic_image = merge_image.crop(sementic_box)
sementic_image.save('sementic/'+'sementic_'+sys.argv[1])
model_sementic_image = sementic_image.crop(model_box)
model_sementic_image.save('sementic/'+'model_sementic_'+sys.argv[1])
sementic_image = Image.open('sementic/'+'sementic_'+sys.argv[1])
gt_sementic_image = sementic_image.crop(gt_box)
gt_sementic_image.save('sementic/'+'gt_sementic_'+sys.argv[1])

merge_image = Image.open(sys.argv[1])
depth_image = merge_image.crop(depth_box)
depth_image.save('depth/'+'depth_'+sys.argv[1])
model_depth_image = depth_image.crop(model_box)
model_depth_image.save('depth/'+'model_depth_'+sys.argv[1])
depth_image = Image.open('depth/'+'depth_'+sys.argv[1])
gt_depth_image = depth_image.crop(gt_box)
gt_depth_image.save('depth/'+'gt_depth_'+sys.argv[1])

merge_image = Image.open(sys.argv[1])
normal_image = merge_image.crop(normal_box)
normal_image.save('normal/'+'normal_'+sys.argv[1])
model_normal_image = normal_image.crop(model_box)
model_normal_image.save('normal/'+'model_normal_'+sys.argv[1])
normal_image = Image.open('normal/'+'normal_'+sys.argv[1])
gt_normal_image = normal_image.crop(gt_box)
gt_normal_image.save('normal/'+'gt_normal_'+sys.argv[1])

# compute losses
# rgb loss
width, height = model_rendering_image.size
rgb_losses = []
rgb_gt = []
for x in range(width):
    for y in range(height):
        model_r, model_g, model_b = model_rendering_image.getpixel((x, y))
        gt_r, gt_g, gt_b = gt_rendering_image.getpixel((x, y))
        rgb_losses.append(np.sqrt((model_r-gt_r)**2+(model_g-gt_g)**2+(model_b-gt_b)**2))
        rgb_gt.append((gt_r, gt_g, gt_b))

# normal loss
width, height = model_normal_image.size
normal_losses = []
normal_gt = []
for x in range(width):
    for y in range(height):
        model_r, model_g, model_b = model_normal_image.getpixel((x, y))
        gt_r, gt_g, gt_b = gt_normal_image.getpixel((x, y))
        model_normal = np.array([model_r/255, model_g/255, model_b/255])
        gt_normal = np.array([gt_r/255, gt_g/255, gt_b/255])
        normal_gt.append(gt_normal)
        normal_losses.append(model_normal.dot(gt_normal) / (np.linalg.norm(model_normal) * np.linalg.norm(model_normal)) )
        

# depth loss
width, height = model_depth_image.size
depth_losses = []
depth_gt = []
for x in range(width):
    for y in range(height):
        model_r, model_g, model_b = model_depth_image.getpixel((x, y))
        model_relative_depth = 0.2989 * model_r + 0.5870 * model_g + 0.1140 * model_b
        gt_r, gt_g, gt_b = gt_depth_image.getpixel((x, y))
        gt_relative_depth = 0.2989 * gt_r + 0.5870 * gt_g + 0.1140 * gt_b
        depth_losses.append(np.abs(model_relative_depth-gt_relative_depth))
        depth_gt.append(gt_relative_depth)

# draw the pic 'bout depth
plt.scatter(depth_gt, rgb_losses, s=0.1)
plt.savefig('./depth-rgb.png')
plt.clf()

plt.scatter(depth_gt, normal_losses, s=0.1)
plt.savefig('./depth-normal.png')
plt.clf()

plt.scatter(depth_gt, depth_losses, s=0.1)
plt.savefig('./depth-depth.png')
plt.clf()

# compute the correlations
depth_rgb_correlations = pearsonr(depth_gt, rgb_losses)[0]
depth_normal_correlations = pearsonr(depth_gt, normal_losses)[0]
depth_depth_correlations = pearsonr(depth_gt, depth_losses)[0]

print("depth_rgb_correlations", depth_rgb_correlations)
print("depth_normal_correlations", depth_normal_correlations)
print("depth_depth_correlations", depth_depth_correlations)

# define the gradient of a grid of 3d vectors
def gradient(vectors):
    gradients = np.zeros(len(vectors))
    displacements = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    length = int(np.sqrt(len(vectors)))
    for i in range(length):
        for j in range(length):
            point = vectors[i*length + j]
            max_dif = 0
            for d in displacements:
                x, y = i+d[0], j+d[1]
                if(0<=x<length and 0<=y<length):
                    new_point = vectors[x*length + y]
                    dif = np.sqrt( (new_point[0]-point[0])**2+(new_point[1]-point[1])**2+(new_point[2]-point[2])**2 ) / np.sqrt(d[0]**2+d[1]**2)
                    if dif > max_dif: max_dif = dif
            gradients[i*length + j] = max_dif
    return gradients

# draw the pic 'bout rgb and normal
rgb_gradient = gradient(rgb_gt)
normal_gradient = gradient(normal_gt)
plt.scatter(rgb_gradient, rgb_losses, s=0.1)
plt.savefig('./rgb-rgb.png')
plt.clf()

plt.scatter(rgb_gradient, normal_losses, s=0.1)
plt.savefig('./rgb-normal.png')
plt.clf()

plt.scatter(rgb_gradient, depth_losses, s=0.1)
plt.savefig('./rgb-depth.png')
plt.clf()

plt.scatter(normal_gradient, rgb_losses, s=0.1)
plt.savefig('./normal-rgb.png')
plt.clf()

plt.scatter(normal_gradient, normal_losses, s=0.1)
plt.savefig('./normal-normal.png')
plt.clf()

plt.scatter(normal_gradient, depth_losses, s=0.1)
plt.savefig('./normal-depth.png')
plt.clf()

rgb_rgb_correlations = pearsonr(rgb_gradient, rgb_losses)[0]
rgb_normal_correlations = pearsonr(rgb_gradient, normal_losses)[0]
rgb_depth_correlations = pearsonr(rgb_gradient, depth_losses)[0]

print("rgb_rgb_correlations", rgb_rgb_correlations)
print("rgb_normal_correlations", rgb_normal_correlations)
print("rgb_depth_correlations", rgb_depth_correlations)

normal_rgb_correlations = pearsonr(normal_gradient, rgb_losses)[0]
normal_normal_correlations = pearsonr(normal_gradient, normal_losses)[0]
normal_depth_correlations = pearsonr(normal_gradient, depth_losses)[0]

print("normal_rgb_correlations", normal_rgb_correlations)
print("normal_normal_correlations", normal_normal_correlations)
print("normal_depth_correlations", normal_depth_correlations)

rgb_losses_matrix = []
normal_losses_matrix = []
depth_losses_matrix = []
rgb_losses_max = max(rgb_losses)
normal_losses_max = max(normal_losses)
depth_losses_max = max(depth_losses)

for i in range(height):
    rgb_row = []
    normal_row = []
    depth_row = []
    for j in range(width):
        rgb_row.append(rgb_losses[height*i+j]/rgb_losses_max)
        normal_row.append(normal_losses[height*i+j]/normal_losses_max)
        depth_row.append(depth_losses[height*i+j]/depth_losses_max)
    rgb_losses_matrix.append(rgb_row)
    normal_losses_matrix.append(normal_row)
    depth_losses_matrix.append(depth_row)

plt.imsave('./rgb_loss.png', np.array(rgb_losses_matrix), cmap='gray')
plt.imsave('./normal_loss.png', np.array(normal_losses_matrix), cmap='gray')
plt.imsave('./depth_loss.png', np.array(depth_losses_matrix), cmap='gray')

plt.scatter(rgb_losses, normal_losses, s=0.1)
plt.savefig('./loss-rgb-normal.png')
plt.clf()

plt.scatter(depth_losses, normal_losses, s=0.1)
plt.savefig('./loss-depth-normal.png')
plt.clf()

plt.scatter(rgb_losses, depth_losses, s=0.1)
plt.savefig('./loss-rgb-depth.png')
plt.clf()

plt.hist(depth_gt, density=True, bins=30)
plt.savefig('./depth-hist.png')