#cython: cdivision=True, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, exp
from cython cimport cdivision

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t



cdef double Hubble_LCDM(double H0, double omega_m, double z):
    return H0*sqrt((omega_m*pow(1+z,3)) + (1-omega_m))



@cdivision(True)
cpdef double inversehubbleparameter_LCDM(double z,double H0, double omega_m):
    return 1/Hubble_LCDM(H0, omega_m, z)






cdef double Hubble_wCDM(double H0, double omega_m, double w, double z):
    cdef double zpower = 3*(1 + w)
    return H0*sqrt(((omega_m)*pow(1+z,3)) +((1-omega_m)*pow(1+z,zpower)))



@cdivision(True)
cpdef double inversehubbleparameter_wCDM(double z,double H0, double omega_m, double w):
    return 1/Hubble_wCDM(H0, omega_m,w, z)


cpdef chisq1_LCDM(double H0, double omega_m, double[::1] z, double[::1] H_data,double[::1] Herr):
    cdef Py_ssize_t i, n = z.shape[0]   
    cdef double[::1] Hvals = np.empty(n, dtype=DTYPE)
    cdef double chisqvals = 0
    cdef double residual
    for i in range(n):
        Hvals[i] = (Hubble_LCDM(H0,omega_m,z[i]))
        residual = (Hvals[i]-H_data[i])/Herr[i]
        chisqvals+= pow(residual,2)
    return chisqvals

cpdef chisq1_wCDM(double H0, double omega_m, double w, double[::1] z, double[::1] H_data,double[::1] Herr):
    cdef Py_ssize_t i, n = z.shape[0]   
    cdef double[::1] Hvals = np.empty(n, dtype=DTYPE)
    cdef double chisqvals = 0
    cdef double residual
    for i in range(n):
        Hvals[i] = (Hubble_wCDM(H0,omega_m, w,z[i]))
        residual = (Hvals[i]-H_data[i])/Herr[i]
        chisqvals+= pow(residual,2)
    return chisqvals


@cdivision(True)
cdef chisquare(double y_true, double y_pred, double y_err):
    cdef double residual = (y_true-y_pred)/y_err
    return pow(residual,2)

cpdef chisq_gen(double[::1] y_true, double[::1] y_pred, double[::1] y_err):
    cdef Py_ssize_t i, n = y_true.shape[0]
    cdef double[::1] chisqvals = np.empty(n, dtype=DTYPE)
    for i in range(n):
        chisqvals[i] = (chisquare(y_true[i],y_pred[i],y_err[i]))
    return np.asarray(chisqvals)