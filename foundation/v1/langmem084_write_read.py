"""Partial independent list reference for084 operators; no trained-source proof."""
import math

def byte_code(byte):
    if type(byte) is not int or not 0<=byte<256:raise ValueError('byte')
    return tuple((1 if (byte>>i)&1 else -1)/math.sqrt(8) for i in range(8))

def read(matrix,key):return tuple(sum(matrix[i][j]*key[j] for j in range(8)) for i in range(8))

def source_update(matrix,key,value,rho,beta,epsilon):
    prior_read=read(matrix,key)
    return tuple(tuple(rho*matrix[i][j]-rho*beta*prior_read[i]*key[j]+epsilon*value[i]*key[j] for j in range(8)) for i in range(8))

def overwrite(matrix,key,value,rho):
    prior_read=read(matrix,key)
    return tuple(tuple(rho*matrix[i][j]+(value[i]-rho*prior_read[i])*key[j] for j in range(8)) for i in range(8))
