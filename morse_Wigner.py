import numpy as np
import matplotlib.pyplot as plt
###### Options graphiques
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('font',**{'size':12})

##########################
#Units are Hartree atomic units (hbar=1) 

##################################################################
#### Variables definition
##################################################################
dx=0.005
xmax=3.0
xmin=-1.0
xvec = np.arange(xmin,xmax,dx)
nx=np.size(xvec)


dp=0.2
Np=int(np.pi/(dx*dp))              
dp=np.pi/(Np*dx)          ##redefining dp more precisely from NP that has been rounded
pmax=Np*dp
pvec=np.fft.fftshift(np.fft.fftfreq(Np,dp))
Wig=np.zeros([nx,Np])
ftvec=np.zeros(Np)


Temp=300.0
kT=Temp*3.166815e-6              #kBT in Hartree
nmax=10

######### Morse potential:
D=0.02
a=3.0
m=1836
def morse(x):
    return D*(1-np.exp(-a*x))**2


Vvec=morse(xvec)
E0=a*np.sqrt(2*D/m); print('E0='+str(E0))

############# Hamiltonian and diagonalisation
H=np.zeros([nx,nx])
H[0,0]=1./(m*dx**2)+Vvec[0]
H[0,1]=1./(2*m*dx**2)
for i in range(1,nx-1):
        H[i,i]=1./(m*dx**2)+Vvec[i]
        H[i,i+1]=-1./(2*m*dx**2)
        H[i,i-1]=-1./(2*m*dx**2)

H[nx-1,nx-1]=1./(m*dx**2)+Vvec[nx-1]
H[nx-1,nx-2]=1./(2*m*dx**2)

ener_full, eigen_full=np.linalg.eigh(H)
ener=ener_full[0:nmax]
eigen=eigen_full[0:nx,0:nmax]/np.sqrt(dx)
proba=np.exp(-ener/kT)
Zpart=np.sum(proba)
proba=proba/Zpart
print('Maximum probability taken into account: '+str(proba[nmax-1]))
######Probability density:
density=np.zeros(nx)
for i in range(nx):
    density[i]=np.sum(proba*eigen[i,:]**2)


#######Computing the Wigner function via Fourier transform:
for i in range(nmax):
    for j in range(nx):
        ftvec[:]=0.0
        ftvec[0]=eigen[j,i]*eigen[j,i]/2.0
        for k in range(1,Np):
            if ((j+k<nx) and (j-k>=0)):  ftvec[k]=eigen[j+k,i]*eigen[j-k,i]
        Wig[j,:]+=(2*dx/np.pi)*proba[i]*np.real(np.fft.fftshift(np.fft.fft(ftvec)))


        

###############Energy verification
print('Numerical energies (first 5):')
print(ener[0:5])
print('Theoretical energies (first 5):')
print(E0*(0.5+np.arange(5))-(E0*(0.5+np.arange(5)))**2/(4*D))

############plots:
for i in range(5):
    plt.plot(xvec,ener[i]*np.ones(nx),color='grey')
plt.plot(xvec,Vvec); plt.ylim(0,2*D)
plt.plot(xvec,D+density*D/np.max(density),'r')
plt.plot(xvec,D+np.exp(-Vvec/kT)*D/np.max(np.exp(-Vvec/kT)),'k--')
plt.show()


plt.figure(); plt.contour(xvec,pvec,np.transpose(Wig),[0.001,0.01,0.05,0.1,0.2])
plt.show()
