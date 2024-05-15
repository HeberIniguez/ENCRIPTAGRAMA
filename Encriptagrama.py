from datetime import datetime
from mpi4py import MPI
import numpy as np
from PIL import Image
from decimal import *
#@profile
def encripta():
        comm=MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
                imagen = Image.open('original.jpg')
                imagen=np.array(imagen)
                ren = imagen.shape[0]
                col = imagen.shape[1]
                cap = imagen.shape[2]
                n_ren=np.int_(ren/size)
                foto=np.reshape(imagen,(size,n_ren,col,cap))
        else:
                foto=None
        #########################################################
        a= Decimal('0.9')
        b= Decimal('-0.6013')
        c= Decimal('2.0')
        d= Decimal('0.5')
        x0= Decimal('-0.72')
        y0= Decimal('-0.64')
        ret=0
        npx0=8
        np_prec=99
        getcontext().prec=npx0
        while (ret<rank+1):
            x1 = Decimal((x0*x0)-(y0*y0)+(a*x0)+(b*y0))
            y1 = Decimal((2*x0*y0)+(c*x0)+(d*y0))
            x0=x1
            y0=y1
            ret=ret+1
        ######################################################
        foto = comm.scatter(foto,root=0)
        ren_local=foto.shape[0]
        col_local=foto.shape[1]
        cap_local=foto.shape[2]
        vector=ren_local*col_local*cap_local
        foto=np.reshape(foto,(vector,1))
        getcontext().prec=np_prec
        s1=0
        s2=0
        j=0
        while(j<vector):
                x1 = Decimal((x0*x0)-(y0*y0)+(a*x0)+(b*y0))
                y1 = Decimal((2*x0*y0)+(c*x0)+(d*y0))
                x0=x1
                y0=y1
                ##################M1####################
                x1=x1*Decimal(10**np_prec)
                y1=y1*Decimal(10**np_prec)
                x1=Decimal.to_integral(x1)
                y1=Decimal.to_integral(y1)
                if x1<0:
                        x1=x1*(-1)
                if y1<0:
                        y1=y1*(-1)
                s1=int(x1%255)
                s2=int(y1%255)
                ####################################
                foto[j]= foto[j]^s1
                foto[j+1]= foto[j+1]^s2
                j=j+2
        foto=np.reshape(foto,(ren_local,col_local,cap_local))
        newData = comm.gather(foto,root=0)
        if rank == 0:
                foto=np.array(newData)
                foto=np.reshape(foto,(ren,col,cap))
                foto=Image.fromarray(foto)
                foto.save('salida.jpg')
                fin=datetime.now()
                 
if __name__ == '__main__':
    encripta()
