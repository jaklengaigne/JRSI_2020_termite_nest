#250620 Adapted from nasute_sub_3D_15 for JRSI repository


#__author__ = "Giulio Facchini"
#__version__ = "0.15"


import h5py
import numpy as np
#import numba as nb
import scipy.ndimage.filters as scfilt 
#from FD_curvature_sub_2_test import *


#get n digit of a number n
#@nb.jit(nopython=True)
def get_digit(number, n):
    return number // 10**n % 10


#___________________________________Kernel for Gaussian Curvature term________________
#For dp=xp*2 and dm=|xm|/4 one obtain a smooth profile (see KG_Kernel.ipynb notebook)
def KG_kernel(x0,xp,xm,dp,dm):
    if (xp+xm+dp+dm)==0:
        Kg_out=x0
    else:
        Kg_out=1/dm/np.sqrt(2*np.pi)*x0*np.exp(-(x0-xm)**2/dm**2)+(1/dp/np.sqrt(2*np.pi))*x0*np.exp(-(x0-xp)**2/dp**2)
    return Kg_out


#@nb.jit("float64[:,:,:](float64[:,:,:],float64,float64,float64)",nopython=True,parallel=True)
def lap3D(U,dx,dy,dz):
	Nz,Nx,Ny=U.shape
	ddU=np.zeros((Nz,Nx,Ny))		
	ddU[2:-2,2:-2,2:-2]=((U[1:-3,2:-2,2:-2]-2*U[2:-2,2:-2,2:-2]+U[3:-1,2:-2,2:-2])/(dz**2) 
			+(U[2:-2,1:-3,2:-2]-2*U[2:-2,2:-2,2:-2]+U[2:-2,3:-1,2:-2])/(dx**2)
			+(U[2:-2,2:-2,1:-3]-2*U[2:-2,2:-2,2:-2]+U[2:-2,2:-2,3:-1])/(dy**2))                  
	return ddU	


#@nb.jit(nopython=True,parallel=True)
def bilap3D(U,dx,dy,dz):
	Nz,Nx,Ny=U.shape
	ddddU= np.zeros((Nz,Nx,Ny))
	ddddU[2:-2,2:-2,2:-2]=((U[2:-2,0:-4,2:-2]-4*U[2:-2,1:-3,2:-2]+6*U[2:-2,2:-2,2:-2]-4*U[2:-2,3:-1,2:-2]+U[2:-2,4:Nx,2:-2])/(dx**4)
		     +(U[2:-2,2:-2,0:-4]-4*U[2:-2,2:-2,1:-3]+6*U[2:-2,2:-2,2:-2]-4*U[2:-2,2:-2,3:-1]+U[2:-2,2:-2,4:Ny])/(dy**4)
		     +(U[0:-4,2:-2,2:-2]-4*U[1:-3,2:-2,2:-2]+6*U[2:-2,2:-2,2:-2]-4*U[3:-1,2:-2,2:-2]+U[4:Nz,2:-2,2:-2])/(dz**4)
		     +2*(U[2:-2,3:-1,1:-3]+U[2:-2,3:-1,3:-1]+U[2:-2,1:-3,1:-3]+U[2:-2,1:-3,3:-1]-2*U[2:-2,2:-2,3:-1]-2*U[2:-2,2:-2,1:-3]-2*U[2:-2,3:-1,2:-2]-2*U[2:-2,1:-3,2:-2]+4*U[2:-2,2:-2,2:-2])/(dx**2*dy**2)
		     +2*(U[3:-1,2:-2,1:-3]+U[3:-1,2:-2,3:-1]+U[1:-3,2:-2,1:-3]+U[1:-3,2:-2,3:-1]-2*U[2:-2,2:-2,3:-1]-2*U[2:-2,2:-2,1:-3]-2*U[3:-1,2:-2,2:-2]-2*U[1:-3,2:-2,2:-2]+4*U[2:-2,2:-2,2:-2])/(dy**2*dz**2)
		     +2*(U[3:-1,1:-3,2:-2]+U[3:-1,3:-1,2:-2]+U[1:-3,1:-3,2:-2]+U[1:-3,3:-1,2:-2]-2*U[2:-2,3:-1,2:-2]-2*U[2:-2,1:-3,2:-2]-2*U[3:-1,2:-2,2:-2]-2*U[1:-3,2:-2,2:-2]+4*U[2:-2,2:-2,2:-2])/(dx**2*dz**2) 		
		     )
	return ddddU


#@nb.jit(nopython=True,parallel=True)
def grad_3D(F,dx,dy,dz):
	Nz,Nx,Ny = F.shape
	Fx=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fy=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fz=np.zeros((Nz,Nx,Ny),dtype=np.float64)

	Fz[2:-2,2:-2,2:-2]=(-F[1:-3,2:-2,2:-2]+F[3:-1,2:-2,2:-2])/dz/2. 		
	Fx[2:-2,2:-2,2:-2]=(-F[2:-2,1:-3,2:-2]+F[2:-2,3:-1,2:-2])/dx/2.
	Fy[2:-2,2:-2,2:-2]=(-F[2:-2,2:-2,1:-3]+F[2:-2,2:-2,3:-1])/dy/2.
	return Fx,Fy,Fz

#@nb.jit(nopython=True,parallel=True)
def Hes_3D(F,dx,dy,dz):
	Nz,Nx,Ny = F.shape
	Fxx=np.zeros((Nz,Nx,Ny))
	Fzz=np.zeros((Nz,Nx,Ny))
	Fyy=np.zeros((Nz,Nx,Ny))
	Fyx=np.zeros((Nz,Nx,Ny))
	Fzx=np.zeros((Nz,Nx,Ny))
	Fzy=np.zeros((Nz,Nx,Ny))

	Fzz[2:-2,2:-2,2:-2]=(F[1:-3,2:-2,2:-2]-2*F[2:-2,2:-2,2:-2]+F[3:-1,2:-2,2:-2])/(dz**2) 
	Fxx[2:-2,2:-2,2:-2]=(F[2:-2,1:-3,2:-2]-2*F[2:-2,2:-2,2:-2]+F[2:-2,3:-1,2:-2])/(dx**2)
	Fyy[2:-2,2:-2,2:-2]=(F[2:-2,2:-2,1:-3]-2*F[2:-2,2:-2,2:-2]+F[2:-2,2:-2,3:-1])/(dy**2)  
                    
	Fzx[2:-2,2:-2,2:-2]=(F[1:-3,1:-3,2:-2]+F[3:-1,3:-1,2:-2]-F[3:-1,1:-3,2:-2]-F[1:-3,3:-1,2:-2])/dz/dx/4.	#Attention to cross derivatives
	Fzy[2:-2,2:-2,2:-2]=(F[1:-3,2:-2,1:-3]+F[3:-1,2:-2,3:-1]-F[3:-1,2:-2,1:-3]-F[1:-3,2:-2,3:-1])/dz/dy/4. 		
	Fyx[2:-2,2:-2,2:-2]=(F[2:-2,1:-3,1:-3]+F[2:-2,3:-1,3:-1]-F[2:-2,1:-3,3:-1]-F[2:-2,3:-1,1:-3])/dx/dy/4. 	

	return Fxx+0.,Fyy+0.,Fzz+0.#,Fzx,Fzy,Fyx

#@nb.jit(nopython=True,parallel=True)
def preKM_3D(F,dx,dy,dz):
	#creating alias  
	Nz,Nx,Ny = F.shape
	P_1= np.zeros((Nz,Nx,Ny))		
	P_2= np.zeros((Nz,Nx,Ny))
	P_3= np.zeros((Nz,Nx,Ny))		#square modulus of the gradient
	P_0 = np.zeros((3,Nz,Nx,Ny),dtype=np.float64)
	Fx=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fy=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fz=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fxx=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fzz=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fyy=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fyx=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fzx=np.zeros((Nz,Nx,Ny),dtype=np.float64)
	Fzy=np.zeros((Nz,Nx,Ny),dtype=np.float64)

	Fzz[2:-2,2:-2,2:-2]=(F[1:-3,2:-2,2:-2]-2*F[2:-2,2:-2,2:-2]+F[3:-1,2:-2,2:-2])/(dz**2) 
	Fz[2:-2,2:-2,2:-2]=(-F[1:-3,2:-2,2:-2]+F[3:-1,2:-2,2:-2])/dz/2. 		

	Fxx[2:-2,2:-2,2:-2]=(F[2:-2,1:-3,2:-2]-2*F[2:-2,2:-2,2:-2]+F[2:-2,3:-1,2:-2])/(dx**2)
	Fx[2:-2,2:-2,2:-2]=(-F[2:-2,1:-3,2:-2]+F[2:-2,3:-1,2:-2])/dx/2.

	Fyy[2:-2,2:-2,2:-2]=(F[2:-2,2:-2,1:-3]-2*F[2:-2,2:-2,2:-2]+F[2:-2,2:-2,3:-1])/(dy**2)                      
	Fy[2:-2,2:-2,2:-2]=(-F[2:-2,2:-2,1:-3]+F[2:-2,2:-2,3:-1])/dy/2.
	Fzx[2:-2,2:-2,2:-2]=(F[1:-3,1:-3,2:-2]+F[3:-1,3:-1,2:-2]-F[3:-1,1:-3,2:-2]-F[1:-3,3:-1,2:-2])/dz/dx/4.	#Attention to cross derivatives
	Fzy[2:-2,2:-2,2:-2]=(F[1:-3,2:-2,1:-3]+F[3:-1,2:-2,3:-1]-F[3:-1,2:-2,1:-3]-F[1:-3,2:-2,3:-1])/dz/dy/4. 		
	Fyx[2:-2,2:-2,2:-2]=(F[2:-2,1:-3,1:-3]+F[2:-2,3:-1,3:-1]-F[2:-2,1:-3,3:-1]-F[2:-2,3:-1,1:-3])/dx/dy/4. 	

	#Fx,Fy,Fz=grad_3D(F,dx,dy,dz)
	
	#    Fxx,Fyy,Fzz,Fzx,Fzy,Fyx=Hes_3D(F,dx,dy,dz)
	P_0[0,:,:,:]=Fz[:,:,:]*Fzz[:,:,:]+Fx[:,:,:]*Fzx[:,:,:]+Fy[:,:,:]*Fzy[:,:,:]   #gradF*HF
	P_0[1,:,:,:]=Fz[:,:,:]*Fzx[:,:,:]+Fx[:,:,:]*Fxx[:,:,:]+Fy[:,:,:]*Fyx[:,:,:]   #gradF*HF
	P_0[2,:,:,:]=Fz[:,:,:]*Fzy[:,:,:]+Fx[:,:,:]*Fyx[:,:,:]+Fy[:,:,:]*Fyy[:,:,:]   #gradF*HF

	P_1[:,:,:]=P_0[0,:,:,:]*Fz[:,:,:]+P_0[1,:,:,:]*Fx[:,:,:]+P_0[2,:,:,:]*Fy[:,:,:]   #gradF*HF  	
	P_3[:,:,:]=Fx[:,:,:]*Fx[:,:,:]+Fy[:,:,:]*Fy[:,:,:]+Fz[:,:,:]*Fz[:,:,:]   #gradF*HF  	
	P_2[:,:,:]=P_3[:,:,:]*(Fxx[:,:,:]+Fyy[:,:,:]+Fzz[:,:,:])

#______very interesting if I add a few lines to this script the code slow down by a factor 2! 
	return P_1 +0.,P_2+0.,P_3+0.#P_1,P_2,P_3 #P_1# 
#	return P_0

#@nb.jit(nopython=True,parallel=True)
def Timestep(U,nn,eq_pars,eq_flags):
	dt=eq_pars[0]; dx=eq_pars[1]; dy=eq_pars[2]; dz=eq_pars[3]; HD=eq_pars[4]; AD=eq_pars[5]; D=eq_pars[6] 
	ADA=eq_pars[7]; S=eq_pars[8]; a=eq_pars[9]; b=eq_pars[10];
	periodic_flag=eq_flags[0];laz_flag=eq_flags[1];source_flag=eq_flags[2];jul_flag=eq_flags[3];thin_flag=eq_flags[4];

	Nz,Nx,Ny=U.shape
	U_new = np.zeros((Nz,Nx,Ny),dtype=np.float64)   #scalar field matrix
	dU = np.zeros((Nz,Nx,Ny),dtype=np.float64)   #scalar field matrix
	ddU = np.zeros((Nz,Nx,Ny),dtype=np.float64)   #scalar field matrix
	KM = np.zeros((Nz,Nx,Ny),dtype=np.float64)	#mean curvature matrix	
	ddddU=bilap3D(U,dx,dy,dz)
#__________computing mean curvature_________________
	eps=ADA
	if thin_flag==1:
		P_1,P_2,P_3=preKM_3D(U,dx,dy,dz);
		KM[2:-2,2:-2,2:-2]=(P_1[2:-2,2:-2,2:-2]-P_2[2:-2,2:-2,2:-2])/2./(P_3[2:-2,2:-2,2:-2]**1.5+eps) #iI cannot include thi line in the preKM_3D function because numba becomes 2 times slower.
		ddUA=-KM
	else:
		FFxx,FFyy,FFzz=Hes_3D(U,dx,dy,dz)
		ddU=lap3D(U,dx,dy,dz)#FFxx+FFyy+FFzz #
		ddUA=ddU		
	if (D!=0) and (thin_flag==1):
		Fxx,Fyy,Fzz=Hes_3D(U,dx,dy,dz)
		#Fxx,Fyy,Fzz,Fzx,Fzy,Fyx=Hes_3D(U_data,dx,dy,dz)  #____If you also need Gaussian curvature
		ddU=Fxx+Fyy+Fzz #lap3D(U,dx,dy,dz)
	else:
		ddU=np.zeros((Nz,Nx,Ny))


	ddU=ddUA
    
    #noisy prefactor
	dU[2:-2,2:-2,2:-2]=nn[2:-2,2:-2,2:-2] * dt * (np.sign(U[2:-2,2:-2,2:-2]) * np.abs(U[2:-2,2:-2,2:-2])**a * np.sign(1-U[2:-2,2:-2,2:-2]) * np.abs(1-U[2:-2,2:-2,2:-2])**b * (-AD * ddUA[2:-2,2:-2,2:-2] - HD * ddddU[2:-2,2:-2,2:-2]*(1-laz_flag))-laz_flag*HD*ddddU[2:-2,2:-2,2:-2]+D*ddU[2:-2,2:-2,2:-2])	
	U_new[2:-2,2:-2,2:-2]=U[2:-2,2:-2,2:-2] + dU[2:-2,2:-2,2:-2]

	#imposing periodic boundary conditions 
	if periodic_flag>=100 and periodic_flag<10000:
		xb=get_digit(periodic_flag,0);
		xt=get_digit(periodic_flag,1);
		U_new[0:2,:,:]=U_new[-4:-2,:,:] 	#pbc on z  
		U_new[-2:Nz,:,:]=U_new[2:4,:,:]
		if xb<=1:			#dirichlet on x bot	
			U_new[:,0:2,:]=xb					        
		else:				#copy x bot from previous step
			U_new[:,:xb,:]=U[:,:xb,:]					
		if xt<=1:			#dirichlet on x top	
			U_new[:,-2:Nx,:]=xt
		else:				#copy x bot from previous step
			U_new[:,-xt:Nx,:]=U[:,-xt:Nx,:]	
		U_new[:,:,0:2]=U_new[:,:,-4:-2]		#pbc on y
		U_new[:,:,-2:Ny]=U_new[:,:,2:4]		
	elif periodic_flag>=10000 and periodic_flag<1000000:
		xb=get_digit(periodic_flag,0);
		xt=get_digit(periodic_flag,1);
		yb=get_digit(periodic_flag,2);
		yt=get_digit(periodic_flag,3);

		U_new[0:2,:,:]=U_new[-4:-2,:,:] 	#pbc on z  
		U_new[-2:Nz,:,:]=U_new[2:4,:,:]
		if xb<=1:			#dirichlet on x bot	
			U_new[:,0:2,:]=xb					        
		else:				#copy x bot from previous step
			U_new[:,:xb,:]=U[:,:xb,:]					
		if xt<=1:			#dirichlet on x top	
			U_new[:,-2:Nx,:]=xt
		else:				#copy x bot from previous step
			U_new[:,-xt:Nx,:]=U[:,-xt:Nx,:]	
		if yb<=1:			#dirichlet on y bot	
			U_new[:,:,:2]=yb	
		else:				#copy y bot from previous step
			U_new[:,:,:yb]=U[:,:,:yb]	
		if yt<=1:			#dirichlet on y top	
			U_new[:,:,-2:Ny]=yt				
		else:				#copy y top from previous step
			U_new[:,:,-yt:Ny]=U[:,:,-yt:Ny]				


	elif periodic_flag>=1000000:
		xb=get_digit(periodic_flag,0);
		xt=get_digit(periodic_flag,1);
		yb=get_digit(periodic_flag,2);
		yt=get_digit(periodic_flag,3);
		zb=get_digit(periodic_flag,4);
		zt=get_digit(periodic_flag,5);

		if zb<=1:			#dirichlet on z bot	
			U_new[:2,:,:]=zb 	 
		else:				#copy z bot from previous step
			U_new[:zb,:,:]=U[:zb,:,:] 	 
		if zt<=1:			#dirichlet on z top	
			U_new[-2:Nz,:,:]=zt		
		else:				#copy z top from previous step
			U_new[-zt:Nz,:,:]=U[-zt:Nz,:,:]		
		if xb<=1:			#dirichlet on x bot	
			U_new[:,:2,:]=xb					        
		else:				#copy x bot from previous step
			U_new[:,:xb,:]=U[:,:xb,:]					
		if xt<=1:			#dirichlet on x top	
			U_new[:,-2:Nx,:]=xt
		else:				#copy x bot from previous step
			U_new[:,-xt:Nx,:]=U[:,-xt:Nx,:]	
		if yb<=1:			#dirichlet on y bot	
			U_new[:,:,:2]=yb	
		else:				#copy y bot from previous step
			U_new[:,:,:yb]=U[:,:,:yb]	
		if yt<=1:			#dirichlet on y top	
			U_new[:,:,-2:Ny]=yt				
		else:				#copy y top from previous step
			U_new[:,:,-yt:Ny]=U[:,:,-yt:Ny]				
	else:
		U_new[0:2,:,:]=U_new[-4:-2,:,:] 	#pbc on z  #This is new in version 11+, previously U_new[..=U[..
		U_new[-2:Nz,:,:]=U_new[2:4,:,:]		
		U_new[:,0:2,:]=U_new[:,-4:-2,:]		#pbc on x 
		U_new[:,-2:Nx,:]=U_new[:,2:4,:]		
		U_new[:,:,0:2]=U_new[:,:,-4:-2]		#pbc on y
		U_new[:,:,-2:Ny]=U_new[:,:,2:4]		
		
	#MEMO FOR PERIODIC FLAG type (1)10 <->   (1)1101<-> **** else ::::     :periodic _=0. *=1.
#					:  :          	    _  *      :  :
#					****	            ****      ::::					  
#	print(nb.config.NUMBA_NUM_THREADS)
	return U_new,dU

'''
standard alias from main program
eq_flags=[periodic_flag,laz_flag,source_flag,jul_flag,thin_flag]
eq_pars=[dt, dx, dy, dz, HD, AD, D, ADA, S, a, b]
save_pars=[save_cad, iter_slice, max_record]
grid_pars=[[Lx,Ly,Lz],[Nx,Ny,Nz],Noise,noise_factor]
'''


def h5_slice_3D(sl_name,x,y,z,U_data,dU,eq_pars,eq_flags,grid_pars,save_pars,save_path):
#creating alias		
	dt=eq_pars[0]; dx=eq_pars[1]; dy=eq_pars[2]; dz=eq_pars[3]; HD=eq_pars[4]; AD=eq_pars[5]; D=eq_pars[6];
	ADA=eq_pars[7]; S=eq_pars[8]; a=eq_pars[9]; b=eq_pars[10]; ADx=eq_pars[11]; ADy=eq_pars[12]; ADz=eq_pars[13]
	save_cad=save_pars[0];iter_slice=save_pars[1]; max_record=save_pars[2];  print_factor=save_pars[3];
	periodic_flag=eq_flags[0];laz_flag=eq_flags[1];source_flag=eq_flags[2];jul_flag=eq_flags[3];thin_flag=eq_flags[4];noise_flag=eq_flags[5];
	ini_flag=eq_flags[6];	restart_flag=eq_flags[7];		
	Lx=grid_pars[0][0];Ly=grid_pars[0][1];Lz=grid_pars[0][2];Nx=grid_pars[1][0];Ny=grid_pars[1][1];Nz=grid_pars[1][2];Noise=grid_pars[2];noise_factor=grid_pars[3];	
	
	f_h5 = h5py.File(save_path + '/'+sl_name, 'w')
	par_set=f_h5.create_group("parameters")    #Here I store physical parameters of the equation and Initial Conditions
	flag_set=f_h5.create_group("flags")    #Here I store physical parameters of the equation and Initial Conditions
	cad_set=f_h5.create_group("cadence")       #Here I store cadence in saving snapshots and hdf5	
#	par_set.create_dataset("par_list", data=par_sim[:][1])
	field_set=f_h5.create_group("fields")      #Here I store physical fields and their derivatives 
	grid_set=f_h5.create_group("grid")        #Here I store data about the simulating grid
	
	p_dset=field_set.create_dataset("p", (Nz,Nx,Ny), dtype='f8',data=U_data, maxshape=(Nz,Nx,Ny))  #main scalar field
	dp_dset=field_set.create_dataset("dp", (Nz,Nx,Ny), dtype='f4',data=dU, maxshape=(Nz,Nx,Ny))  #main scalar field
	nn_dset=field_set.create_dataset("nn", (Nx,Ny), dtype='f4',data=np.zeros((Nx,Ny)), maxshape=(Nx,Ny))       # z-cut of the noise mask applied to the increment
#	dd_dset=field_set.create_dataset("ddp", (1,Nz,Nx,Ny), dtype='f8', maxshape=(iter_slice,Nz,Nx,Ny))  #main scalar field
	iter_dset=field_set.create_dataset("iter",dtype='i',data=0)  #iteration 
#	p_dset[0,:,:]=U_data[:,:]
#	iter_dset[0]=0
	grid_set.create_dataset("x",(1,Nx-2),dtype='f8',data=x)
	grid_set.create_dataset("y",(1,Ny-2),dtype='f8',data=y)
	grid_set.create_dataset("z",(1,Nz-2),dtype='f8',data=z)
	grid_set.create_dataset("Nx",dtype='i',data=Nx)
	grid_set.create_dataset("Ny",dtype='i',data=Ny)
	grid_set.create_dataset("Nz",dtype='i',data=Nz)
	grid_set.create_dataset("Lx",dtype='f8',data=Lx)
	grid_set.create_dataset("dx",dtype='f8',data=dx)
	grid_set.create_dataset("Ly",dtype='f8',data=Ly)
	grid_set.create_dataset("dy",dtype='f8',data=dy)
	grid_set.create_dataset("Lz",dtype='f8',data=Lz)
	grid_set.create_dataset("dz",dtype='f8',data=dz)
	grid_set.create_dataset("dt",dtype='f8',data=dt)

	par_set.create_dataset("Diffusion",dtype='f8',data=D)
	par_set.create_dataset("AntiDiffusion",dtype='f8',data=AD)
	par_set.create_dataset("HyperDiffusion",dtype='f8',data=HD)
	par_set.create_dataset("Noise",dtype='f8',data=Noise)
	par_set.create_dataset("noise_factor",dtype='f8',data=noise_factor)
	par_set.create_dataset("Source",dtype='f8',data=S)
	par_set.create_dataset("a",dtype='f8',data=a)
	par_set.create_dataset("b",dtype='f8',data=b)
	par_set.create_dataset("ADA",dtype='f8',data=ADA)
	par_set.create_dataset("ADx",dtype='f8',data=ADx)
	par_set.create_dataset("ADy",dtype='f8',data=ADy)
	par_set.create_dataset("ADz",dtype='f8',data=ADz)

	cad_set.create_dataset("save_cad",dtype='i',data=save_cad)
	cad_set.create_dataset("iter_slice",dtype='i',data=iter_slice)
	cad_set.create_dataset("max_record",dtype='i',data=max_record)
	cad_set.create_dataset("print_factor",dtype='i',data=print_factor)


	flag_set.create_dataset("source_flag",dtype='i',data=source_flag)
	flag_set.create_dataset("jul_flag",dtype='i',data=jul_flag)
	flag_set.create_dataset("laz_flag",dtype='i',data=laz_flag)
	flag_set.create_dataset("noise_flag",dtype='i',data=noise_flag)
	flag_set.create_dataset("periodic_flag",dtype='i',data=periodic_flag)
	flag_set.create_dataset("thin_flag",dtype='i',data=thin_flag)
	flag_set.create_dataset("ini_flag",dtype='i',data=ini_flag)
	f_h5.create_dataset("name",data=sl_name)

	return [f_h5, p_dset, dp_dset, iter_dset, nn_dset]

def h5_restart_3D(file_name):
	f = h5py.File( file_name,'r')  #after update <-----giulio
	#I print the HDF5 file field
	print ("%s contains:" %file_name)
	for field in f:
		if field!="name":
			print("->%s"%field)
			for s_field in f[field]:
				if (field=="parameters" or field =='cadence' or field =='flags' or field=='grid') and f[field][s_field].shape==():
					print("______-->%s=%1.4f"%(s_field,f[field][s_field][()]))
				else:
					print("______-->%s"%s_field)
	#creating subgroups			
	eq_flags=[]
	eq_pars=[]
	save_pars=[]
#	U_0=f["fields"]["p"][-1,:,:,:]

#	ddU_0=f["fields"]["ddp"][-1,:,:,:]
	iter_0=f["fields"]["iter"][()]               
	if isinstance(iter_0,np.ndarray):
		iter_0=iter_0[0]               
	Nx=f["grid"]["Nx"][()]
	Ny=f["grid"]["Ny"][()]
	Nz=f["grid"]["Nz"][()]
	U_0=f["fields"]["p"][()]
	if len(U_0.shape)==4:	
		U_0=(f["fields"]["p"][-1,:,:,:]).reshape(Nz,Nx,Ny)
	Lx=f["grid"]["Lx"][()]
	Ly=f["grid"]["Ly"][()]
	Lz=f["grid"]["Lz"][()]

	dx=f["grid"]["dx"][()]
	dy=f["grid"]["dy"][()]
	dz=f["grid"]["dz"][()]
	dt=f["grid"]["dt"][()]
	x=f["grid"]["x"][:].squeeze()
	y=f["grid"]["y"][:].squeeze()
	z=f["grid"]["y"][:].squeeze()

	D=f["parameters"]["Diffusion"][()]
	AD=f["parameters"]["AntiDiffusion"][()]
	HD=f["parameters"]["HyperDiffusion"][()]
	Noise=f["parameters"]["Noise"][()]
	noise_factor=f["parameters"]["noise_factor"][()]
	S=f["parameters"]["Source"][()]
	ADA=f["parameters"]["ADA"][()]
	ADx=f["parameters"]["ADx"][()]
	ADy=f["parameters"]["ADy"][()]
	ADz=f["parameters"]["ADz"][()]
	a=f["parameters"]["a"][()]
	b=f["parameters"]["b"][()]

	save_cad=f["cadence"]["save_cad"][()]
	iter_slice=f["cadence"]["iter_slice"][()]
	max_record=f["cadence"]["max_record"][()]
	print_factor=f["cadence"]["print_factor"][()]

	source_flag=f["flags"]["source_flag"][()]
	laz_flag=f["flags"]["laz_flag"][()]
	jul_flag=f["flags"]["jul_flag"][()]
	noise_flag=f["flags"]["noise_flag"][()]
	periodic_flag=f["flags"]["periodic_flag"][()]
	thin_flag=f["flags"]["thin_flag"][()]
	ini_flag=f["flags"]["ini_flag"][()]
	restart_flag=f["flags"]["ini_flag"][()]

	
	eq_flags=[periodic_flag,laz_flag,source_flag,jul_flag,thin_flag,noise_flag,ini_flag,restart_flag]
	eq_pars=[dt, dx, dy, dz, HD, AD, D, ADA, S, a, b, ADx,ADy,ADz]
	save_pars=[save_cad, iter_slice, max_record, print_factor]
	grid_pars=[[Lx,Ly,Lz],[Nx,Ny,Nz],Noise,noise_factor]
	f.close()	#closing files						 
#	return [f,x,y,z,iter_0,U_0,ddU_0,eq_pars,eq_flags,grid_pars,save_pars]
	return [f,x,y,z,iter_0,U_0,eq_pars,eq_flags,grid_pars,save_pars]

def h5_resize_3D(file_name,sh):	       #to read and resize one arbitrary 3D run 
	print(sh)
	f = h5py.File(file_name,'r')  #read into file
	run_name=file_name
	if ("/name" in f): 
		run_name=f["name"][()]
	else:
		run_name=file_name
	#get the original dimensions of the field in hdf5 file
	Nx=f["grid"]["Nx"][()]
	Ny=f["grid"]["Ny"][()]
	Nz=f["grid"]["Nz"][()]
	U_0=f["fields"]["p"][()]
	if len(U_0.shape)==4:	
		U_0=(f["fields"]["p"][-1,:,:,:]).reshape(Nz,Nx,Ny)
	sh_old=np.copy(sh)*0
	index_s=np.zeros(3,dtype=int)
	index_l=np.zeros(3,dtype=int)
	index_s_old=np.zeros(3,dtype=int)
	sh_old[0]=f["grid"]["Nx"][()]
	sh_old[1]=f["grid"]["Ny"][()]
	sh_old[2]=f["grid"]["Nz"][()]
	for j in range (0,3):	
		index_s[j]=(sh[j]-sh_old[j])//2;			
		index_s[j]=abs(index_s[j])*(index_s[j]>0)	#new start index: 0 if new matrix < old one, 1/2 the diff new and old otherwise
		index_s_old[j]=abs(index_s[j])*(index_s[j]<0)	#old start index: 0 if old matrix < new one, 1/2 the diff old and new otherwise
		if len(sh)>(j+3):
			index_s[j]=index_s[j]*sh[j+3]*2
			index_s_old[j]=index_s_old[j]*sh[j+3]*2
		index_l[j]=min(sh[j],sh_old[j]);		#length in px of the portion to copy, this must be the smallest dimension		
	U_new=np.zeros((sh[2],sh[0],sh[1]))			#prepare new matrix
	U_new[index_s[2]:index_s[2]+index_l[2],index_s[0]:index_s[0]+index_l[0],index_s[1]:index_s[1]+index_l[1]]=U_0[index_s_old[2]:index_s_old[2]+index_l[2],index_s_old[0]:index_s_old[0]+index_l[0],index_s_old[1]:index_s_old[1]+index_l[1]]
	f.close()	#closing files						 
	return [U_new,U_0,run_name]

def disp_via_mask(U_data,mask):
    Nx=U_data.shape[1]
    Ny=U_data.shape[2]
    Nz=U_data.shape[0]
    U_data_disp=np.copy(U_data)
    if (Nz==mask.shape[0]) and(Nx==mask.shape[1]):                 
        for i in range(0,Nz):
            for j in range(0,Nx):
                for k in range(0,Ny):
                    U_data_disp[i,j,k]=U_data[i,j,(k+int(mask[i,j]))%Ny]
    else:
        print('no displacement, mask shape should copy the first 2-dimensions of the input stack')
    return U_data_disp

def float_list(in_list,sep):
    str_list=in_list.split('_')
    out_list=str_list
    for i in range (0, len(str_list)):
        out_list[i]=float(str_list[i])
    return out_list

def Timestep_smooth(U,eq_pars,eq_flags,grid_pars):
    #creating alias
    dt=eq_pars[0]; dx=eq_pars[1]; dy=eq_pars[2]; dz=eq_pars[3]; HD=eq_pars[4]; AD=eq_pars[5]; D=eq_pars[6] 
    ADA=eq_pars[7]; S=eq_pars[8]; a=eq_pars[9]; b=eq_pars[10]; ADx=eq_pars[11]; ADy=eq_pars[12]; ADz=eq_pars[13]
    periodic_flag=eq_flags[0];laz_flag=eq_flags[1];source_flag=eq_flags[2];jul_flag=eq_flags[3];thin_flag=eq_flags[4];
    Nx=grid_pars[1][0];Ny=grid_pars[1][1];Nz=grid_pars[1][2];			
    U_new = np.zeros((Nz,Nx,Ny))
    ddU	= np.zeros((Nz,Nx,Ny))
    ddddU= np.zeros((Nz,Nx,Ny))
    #ADx=ADy=ADz=1.
    ddU[2:-2,2:-2,2:-2]=((U[1:-3,2:-2,2:-2]-2*U[2:-2,2:-2,2:-2]+U[3:-1,2:-2,2:-2])/(dz**2)+ 
			(U[2:-2,1:-3,2:-2]-2*U[2:-2,2:-2,2:-2]+U[2:-2,3:-1,2:-2])/(dx**2)+
			(U[2:-2,2:-2,1:-3]-2*U[2:-2,2:-2,2:-2]+U[2:-2,2:-2,3:-1])/(dy**2))                      #by lines equivalent
#
    ddddU[2:-2,2:-2,2:-2]=((U[2:-2,0:-4,2:-2]-4*U[2:-2,1:-3,2:-2]+6*U[2:-2,2:-2,2:-2]-4*U[2:-2,3:-1,2:-2]+U[2:-2,4:Nx,2:-2])/(dx**4)
    	             +(U[2:-2,2:-2,0:-4]-4*U[2:-2,2:-2,1:-3]+6*U[2:-2,2:-2,2:-2]-4*U[2:-2,2:-2,3:-1]+U[2:-2,2:-2,4:Ny])/(dy**4)
    	             +(U[0:-4,2:-2,2:-2]-4*U[1:-3,2:-2,2:-2]+6*U[2:-2,2:-2,2:-2]-4*U[3:-1,2:-2,2:-2]+U[4:Nz,2:-2,2:-2])/(dz**4)
                     +2*(U[2:-2,3:-1,1:-3]+U[2:-2,3:-1,3:-1]+U[2:-2,1:-3,1:-3]+U[2:-2,1:-3,3:-1]-2*U[2:-2,2:-2,3:-1]-2*U[2:-2,2:-2,1:-3]-2*U[2:-2,3:-1,2:-2]-2*U[2:-2,1:-3,2:-2]+4*U[2:-2,2:-2,2:-2])/(dx**2*dy**2)
                     +2*(U[3:-1,2:-2,1:-3]+U[3:-1,2:-2,3:-1]+U[1:-3,2:-2,1:-3]+U[1:-3,2:-2,3:-1]-2*U[2:-2,2:-2,3:-1]-2*U[2:-2,2:-2,1:-3]-2*U[3:-1,2:-2,2:-2]-2*U[1:-3,2:-2,2:-2]+4*U[2:-2,2:-2,2:-2])/(dy**2*dz**2)
                     +2*(U[3:-1,1:-3,2:-2]+U[3:-1,3:-1,2:-2]+U[1:-3,1:-3,2:-2]+U[1:-3,3:-1,2:-2]-2*U[2:-2,3:-1,2:-2]-2*U[2:-2,1:-3,2:-2]-2*U[3:-1,2:-2,2:-2]-2*U[1:-3,2:-2,2:-2]+4*U[2:-2,2:-2,2:-2])/(dx**2*dz**2) 		
                     )

    U_new[2:-2,2:-2,2:-2]=U[2:-2,2:-2,2:-2]+1.*dt*(HD*ddU[2:-2,2:-2,2:-2]-0*HD*ddddU[2:-2,2:-2,2:-2])
    if periodic_flag==1:	        
        U_new[0:2,:,:]=U[-4:-2,:,:] 
        U_new[-2:Nz,:,:]=U[2:4,:,:]
        U_new[:,0:2,:]=U[:,-4:-2,:]
        U_new[:,-2:Nx,:]=U[:,2:4,:]
        U_new[:,:,0:2]=U[:,:,-4:-2]
        U_new[:,:,-2:Ny]=U[:,:,2:4]  
    return U_new



