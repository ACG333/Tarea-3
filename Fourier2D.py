import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg

from scipy.fftpack import fft, ifft
from matplotlib.colors import LogNorm

from PIL import Image

#-------------Punto 4.1------------------
#%%


#from matplotlib import pyplot as plt

filename = 'Arboles'

img = Image.open( filename + '.png' )
data = np.array( img, dtype='uint8' )

#print data


#-------------Punto 4.2------------------
#%%
n= data.size
fou = np.fft.fft2(data)

fou_changed = np.fft.fftshift(fou)

plt.figure()
plt.imshow(abs(fou_changed), norm=LogNorm())
plt.colorbar()
plt.title('transformada de Fourier 2D')

plt.savefig('CorralesAlejandro_FT2D.pdf')

#-------------Punto 4.3------------------
#%%

foufiltrada = fou_changed.copy()

for i in range(fou.shape[0]):
    for j in range(fou.shape[1]):
        if(abs(fou_changed[i,j])>100000):
            foufiltrada[i,j]=0


#-------------Punto 4.4------------------
#%%
plt.figure()
plt.imshow(abs(foufiltrada), norm=LogNorm())
plt.colorbar()
plt.title('transformada de Fourier 2D filtrada')


#-------------Punto 4.5------------------
#%%

inversa=np.fft.ifft2(foufiltrada).real

print inversa

plt.figure()
plt.imshow(abs(inversa))
plt.title('imagen corregida')
plt.savefig('CorralesAlejandro_Imagen_filtrada.pdf')

plt.figure()
plt.imshow(data)





