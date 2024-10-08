import os
from glob import glob

f = open('resources/train.txt', 'w')
imagenet_basepath = './dataset/'
imgs = glob(imagenet_basepath + '*/*.bmp')
for img in imgs:
  f.write(img + '\n')
f.close()