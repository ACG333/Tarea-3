import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg
import urllib2
import time

from scipy.fftpack import fft, fftfreq

import pandas as pd

from PIL import Image

#-------------Punto 4.1------------------
#%%


#from matplotlib import pyplot as plt

filename = 'Arboles'

img = Image.open( filename + '.png' )
data = np.array( img, dtype='uint8' )

#print data
#plt.imshow(data)

'''
np.save( filename + '.npy', data)

# visually testing our output
img_array = np.load(filename + '.npy')
plt.imshow(img_array)
'''
#-------------Punto 4.2------------------
#%%
n= data.size
fo = np.fft.fft2(data)



frec= np.fft.fftfreq(n)
print np.shape(data)
print np.shape(fo)
print np.shape(frec)

#print fou

print frec

plt.figure()
plt.plot(abs(fo))

'''
fou_changed = np.fft.fftshift(fou)
frec_changed = np.fft.fftshift(frec)


plt.grid()
#plt.xlim(-0.03,0.03)



#-------------Punto 4.3------------------
#%%

#-------------Punto 4.4------------------

#-------------Punto 4.5------------------

'''