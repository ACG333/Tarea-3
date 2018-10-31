import numpy as np
import matplotlib.pyplot as plt

import numpy.linalg
import urllib2
import time

from scipy.fftpack import fft, fftfreq

import pandas as pd

#-------------Punto 2.1------------------
'''
archivoDescargar = "http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat"
archivoGuardar = "WDBC.dat"

now = time.time()

descarga = urllib2.urlopen(archivoDescargar)

ficheroGuardar=file(archivoGuardar,"w")

ficheroGuardar.write(descarga.read())

ficheroGuardar.close()

elapsed = time.time() - now

print "Descargado el archivo: %s en %0.3fs" % (archivoDescargar,elapsed)
'''
#-------------Punto 2.2------------------


data = np.genfromtxt('WDBC.dat',dtype=float,delimiter=',')

d= data[:,2::]
dataI = d.T

def cov_matrix(dataI):
    n_dim = np.shape(dataI)[1]
    n_points = np.shape(dataI)[0]
    cov = np.zeros([n_points, n_points])
    for i in range(n_points):
        for j in range(n_points):
            mean_linea = np.mean(dataI[i,:])
            mean_linea2 = np.mean(dataI[j,:])
            cov[i,j] = np.sum((dataI[i,:]-mean_linea) * (dataI[j,:]-mean_linea2)) / (n_dim -1)
    return cov
    

covA = cov_matrix(dataI)

#########################################print covA

#-------------Punto 2.3------------------

valsA, vecsA = numpy.linalg.eig(covA)

print valsA

xf=np.linspace(-100,100,valsA.size)


#for p in range(np.shape(dataI)[0]):
#    print "Autovalor", p+1, ":\n", valsA[p], "\n y su Autovector correspondiente es:\n", vecsA[p]


for j in range(valsA.size):
    max(valsA)

#-------------Punto 2.4------------------



#-------------Punto 2.5------------------
'''
plt.figure()
#plt.scatter(dataI[21,:], dataI[12,:], alpha=0.3)
x = np.linspace(-100,100, 10)
#primer autovector
m = vecsA[1,0]/vecsA[0,0]
plt.plot(x, m*x, label='Primer Autovector')

#segundo autovector
m = vecsA[1,1]/vecsA[0,1]
plt.plot(x, m*x, label='Segundo Autovector')

#primer autovector
m = vecsA[1,2]/vecsA[0,2]
plt.plot(x, m*x, label='Tercer Autovector')

#segundo autovector
m = vecsA[1,3]/vecsA[0,3]
plt.plot(x, m*x, label='Cuarto Autovector')

#-------------Punto 2.6------------------

print ("el PCA es util por que si")



'''