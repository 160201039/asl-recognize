from conv import conv_layer
from relu import reLU

import skimage
import numpy
import matplotlib.pyplot as plt
from skimage.color import rgb2gray



l1_filter = numpy.zeros((2,3,3))

l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]]])
l1_filter[1, :, :] = numpy.array([[[1,   1,  1],
                                   [0,   0,  0],
                                   [-1, -1, -1]]])

img=plt.imread('/home/erhan/Masaüstü/39413/00001.jpg')
#img=skimage.data.chelsea()
img=rgb2gray(img)



features=conv_layer(img,l1_filter)
reLUFeatures=reLU(features)




fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(reLUFeatures[:,:,0],cmap=plt.cm.gray)
fig.add_subplot(1,2,2)
plt.imshow(reLUFeatures[:,:,1],cmap=plt.cm.gray)


plt.show()



