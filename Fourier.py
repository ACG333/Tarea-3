import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg
import urllib2
import time

from scipy.fftpack import fft, fftfreq

import pandas as pd


#-------------Punto 3.1------------------

incompletos = np.genfromtxt('incompletos.dat',dtype=float,delimiter=',')
signal = np.genfromtxt('signal.dat',dtype=float,delimiter=',')



#-------------Punto 3.2------------------
x= signal[:,0]
y= signal[:,1]

plt.figure()
plt.plot(x,y)
############################plt.savefig('CorralesAlejandro_signal.pdf')



#-------------Punto 3.3------------------
n= y.size
fou= np.fft.fft(y)


#-------------Punto 3.4------------------

frec= np.fft.fftfreq(n)

fou_changed = np.fft.fftshift(fou)
frec_changed = np.fft.fftshift(frec)

plt.figure()
plt.plot(frec_changed,abs(fou_changed))
plt.grid()
plt.xlim(-0.03,0.03)
##############################plt.savefig('CorralesAlejandro_TF.pdf')


#fft_x_half = (2.0) * fft_x[:half_n]
#freq_half = freq[:half_n]

print n/2
 

#-------------Punto 3.5------------------

fouA= abs(fou)
peaks=[]

for i in range(np.size(fouA)/2):
    dato=fouA[i+1]
    if fouA[i+2]<dato and fouA[i]<dato:
        peaks.append(dato)
print peaks

#-------------Punto 3.6------------------

#-------------Punto 3.7------------------

#-------------Punto 3.8------------------

#-------------Punto 3.9------------------

#-------------Punto 3.10------------------

#-------------Punto 3.11------------------

#-------------Punto 3.12------------------



