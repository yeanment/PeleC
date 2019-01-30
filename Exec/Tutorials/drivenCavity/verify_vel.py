import yt
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
    

#=======================================
#read Ghia and Ghia solution
#=======================================
infile=open("Ghia_soln_Re100",'r')

y_ghia = np.array([])
v_ghia = np.array([])

for line in infile:
    splt=line.split()
    y_ghia=np.append(y_ghia,float(splt[0]))
    v_ghia=np.append(v_ghia,float(splt[1]))

infile.close()
#=======================================

#=======================================
#read Pele solution
#=======================================
ds=yt.load(argv[1])
fieldname="x_velocity"
res = 100
zslice = 0.0625
slc = ds.slice('z',zslice)
frb = slc.to_frb((1,'cm'), res)
y = np.linspace(0,1,res)
fld = np.array(frb[fieldname])[:,res/2]
#=======================================

#=======================================
#Plot solutions
#=======================================
plt.figure()
plt.plot(y,fld/np.max(fld),'k',label="Pele")
plt.plot(y_ghia,v_ghia,'r*',label="Ghia et al.,JCP,48,pp 387-411,1982")
plt.legend(loc="best")

plt.figure()
plt.imshow(np.array(frb[fieldname]),origin="lower")
plt.colorbar()
plt.show()
#=======================================

