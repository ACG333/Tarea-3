import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg
import urllib2
import time

from scipy.fftpack import fft, fftfreq, ifft

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

def Transformada(y):
    cadena=[]
    for i in range(len(y)):
        conteo=0
        for j in range(len(y)):
            conteo=(np.exp(-(2j*i*j*np.pi)/len(y))*y[j])+(conteo)
        cadena.append(conteo)
        
    return np.array(cadena)
    
fou=Transformada(y)


#-------------Punto 3.4------------------

frec= np.fft.fftfreq(n)

fou_changed = np.fft.fftshift(fou)
frec_changed = np.fft.fftshift(frec)

plt.figure()
plt.plot(frec_changed,abs(fou_changed))
plt.grid()
plt.xlim(-0.03,0.03)

##############################plt.savefig('CorralesAlejandro_TF.pdf')


#-------------Punto 3.5------------------

Nuevo=[]
for i in range(len(fou_changed)):
    if(abs(fou_changed[i]/max(fou))>0.4) and(frec_changed[i]>0):
        Nuevo.append(frec_changed[i])

for i in range(len(Nuevo)):
    print "Frecuencia principal", i+1, ":", Nuevo[i]


#-------------Punto 3.6------------------

for i in range(len(fou_changed)):
    if(abs(fou_changed[i])>100):
        fou_changed[i]=0
        
plt.figure()
plt.plot(frec_changed,abs(fou_changed))
plt.grid()
plt.xlim(-0.03,0.03)
##############################plt.savefig('CorralesAlejandro__filtrada.pdf')

#-------------Punto 3.7------------------

print "los datos de 'incompletos.dat' no se pueden imprimir ya que los datos no estan completos y ademas de esto, estan mal espaciados"


#-------------Punto 3.8------------------
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate

x1=incompletos[:,0]
y1=incompletos[:,1]

xplot=np.linspace(min(x1),max(x1),512)

def interpole(x,y):
    f1 = sp.interpolate.interp1d(x, y, kind='quadratic')
    f2 = sp.interpolate.interp1d(x, y, kind='cubic')
    return f1, f2

F0=interpole(x1,y1)[0]
F1=interpole(x1,y1)[1]

F0plot= F0(xplot)
F1plot= F1(xplot)


u1=np.fft.fft(F0plot)
u2=np.fft.fft(F1plot)



#-------------Punto 3.9------------------
#%%

F0=interpole(x,y)[0]
F1=interpole(x,y)[1]

F0plot= F0(xplot)
F1plot= F1(xplot)


plt.figure()
xplot=np.linspace(min(x),max(x),512)

plt.subplot(2,2,1)
plt.scatter(x,y)
plt.xlim(-0.001,0.03)
plt.title('datos originales')

plt.subplot(2,2,2)
plt.plot(xplot,F0plot)
plt.title('cuadratica')

plt.subplot(2,2,3)
plt.plot(xplot,F1plot)
plt.title('cubica')

plt.savefig('CorralesAlejandro_TF_interpola.pdf')



#-------------Punto 3.10------------------

print "Hechas las interpolaciones es facil concluir que la cuadratica aumenta el ruido de los datos originales, mientras que la cubica los reduce eficientemente"

#-------------Punto 3.11------------------



#-------------Punto 3.12------------------



