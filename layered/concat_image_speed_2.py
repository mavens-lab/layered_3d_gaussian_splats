import torch
from os import makedirs
import numpy as np
from PIL import Image
import cv2
import os


render_path = os.path.join('/nfs/turbo/coe-mavens/zxuechen/LightGaussian/result', "speed")
out_path = os.path.join(render_path,'concat_2')
fast_path = "/home/zxuechen/LightGaussian/result/load_path_sorting_ours_fast/ours_5000"
makedirs(out_path,exist_ok=True)
fourcc = cv2.VideoWriter.fourcc(*'DIVX') 
image_dir = "/home/zxuechen/LightGaussian/result/load_path_ours2/ours_5000"
first_image_path = os.path.join(image_dir, '{0:05d}'.format(0) + ".png")
first_image = Image.open(first_image_path)
size = (first_image.width*5, first_image.height)
fps = float(36)
writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)
if not writer.isOpened():
    print("Error: Failed to open video writer.")
else:
    print('Video writer opened successfully.')
    image_dir = "/home/zxuechen/LightGaussian/result/load_path_ours2/ours_5000"
    ii = 0
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.endswith('.png'):
            ours = np.array(Image.open(os.path.join(image_dir,file_name)))
            print(ours.shape)
            fast = np.array(Image.open(os.path.join(fast_path,file_name)))
            print(fast.shape)
            result = np.hstack([ours,fast])
            result = result.astype('uint8')
            Image.fromarray(result).save(os.path.join(out_path,file_name))
            writer.write(result[:,:,::-1])
            ii = ii + 1
    writer.release()