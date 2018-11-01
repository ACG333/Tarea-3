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

xf=np.linspace(-100,100,valsA.size)


#for p in range(np.shape(dataI)[0]):
#    print "Autovalor", p+1, ":\n", valsA[p], "\n y su Autovector correspondiente es:\n", vecsA[p]



#-------------Punto 2.4------------------


#print valsA-----> Los autovalores mas altos estan al principio del array en este caso

PC1=vecsA[0]
PC2=vecsA[1]


print "El componente principal 1 (PC1) es: \n", PC1, " \ny el componente principal 2 (PC2) es: \n", PC2


#-------------Punto 2.5------------------

PC11=[]
PC22=[]

for i in range(len(data)):
    A1=np.dot(d[i],PC1)
    A2=np.dot(d[i],PC2)
    PC11.append(A1)
    PC22.append(A2)

plt.figure()
plt.scatter(PC11, PC22, alpha=0.3)
plt.title("PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

#-------------Punto 2.6------------------

print ("el PCA es util por que se pueden hayar que tan relacionados estan los tumores benignos y los malignos")


