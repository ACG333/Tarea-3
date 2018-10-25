import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import urllib2
import time
'''
#-------------Punto 2.1------------------

archivoDescargar = "http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat"
archivoGuardar = "WDBC.dat"

now = time.time()

descarga = urllib2.urlopen(archivoDescargar)

ficheroGuardar=file(archivoGuardar,"w")

ficheroGuardar.write(descarga.read())

ficheroGuardar.close()

elapsed = time.time() - now

print "Descargado el archivo: %s en %0.3fs" % (archivoDescargar,elapsed)

#-------------Punto 2.2------------------

'''

data = np.genfromtxt('WDBC.dat',dtype=float,delimiter=',')

print data[0,:]

'''
def cov_matrix(data):
    n_dim = np.shape(data)[1]
    n_points = np.shape(data)[0]
    cov = np.ones([n_dim, n_dim])
    for i in range(n_dim):
        for j in range(n_dim):
            mean_i = np.mean(data[:,i])
            mean_j = np.mean(data[:,j])
            cov[i,j] = np.sum((data[:,i]-mean_i) * (data[:,j]-mean_j)) / (n_points -1)
    return cov


dataA = np.loadtxt('WDBC.dat')
covA = cov_matrix(dataA)
valsA, vecsA = numpy.linalg.eig(covA)


'''
