#___250620 This version adapted from nasute_3d_term_numba_7.py for JRSI repository

#__author__ = "Giulio Facchini"
#__version__ = "0.7"




''' 
Demo command line to obtain simulation (b), initial conditions are a nest seed on the bottom.
python3 ./nasute_main_3D_7.py 1 -AD 4. -Nx 256 -Ny 256 -Nz 256 -Lx 128 -Ly 128 -Lz 128 -IC start_field.hdf5 -t 400. -bc 104 -f 0.85 

Demo command line to obtain simulation (a), initial conditions are a volume filled with white noise.
python3 ./nasute_main_3D_7.py 1 -AD 4. -Nx 256 -Ny 256 -Nz 256 -Lx 128 -Ly 128 -Lz 128 -t 400. -bc 1 
'''


from pylab import *
import numpy as np
import datetime
import h5py
import os
#sys.path.append('./subroutines/')
from nasute_sub_3D_15 import *
import scipy.ndimage.filters as scfilt 
import numba as nb
import argparse



#______to avoid annoying bug with negative input which are not an option!!
for i, arg in enumerate(sys.argv):
  if (arg[0] == '-') and arg[1].isdigit(): sys.argv[i] = ' ' + arg


#___BUILD PARSER_________
parser = argparse.ArgumentParser(prefix_chars='-')
parser.add_argument("serial", help="serial number of the run or restart file")
parser.add_argument("-dt",type=float,default=0.0003, help="TimeStep")
parser.add_argument("-HD","--HyperDiffusion",type=float,default=-1., help="Hyper-Diffusion")
parser.add_argument("-AD","--AntiDiffusion", type=float,default=4., help="Anti-Diffusion")
parser.add_argument("-Nx", help="Nx",type=int,default=64)
parser.add_argument("-Ny", help="Ny",type=int,default=64)
parser.add_argument("--log", help="logging cadence num of iterations",type=int,default=100)
parser.add_argument("-Nz", help="Nz",type=int,default=64)
parser.add_argument("-D","--Diffusion", help="Diffusion",type=float,default=0.)
parser.add_argument("-gn","--growth_noise", help="growth noise",default=[0.,0.8,2.],type=float,nargs='*')
parser.add_argument("-Lx", help="Lx",type=float,default=32)
parser.add_argument("-Ly", help="Ly",type=float,default=32)
parser.add_argument("-Lz", help="Lz",type=float,default=32)
parser.add_argument("-N", "--Noise", help="Noise",default=1.,type=float)
parser.add_argument("-nf","--noise_factor", help="noise_factor",default=2.,type=float)
parser.add_argument("-t","--sim_time", help="sim_time",default=100.,type=float)
parser.add_argument("-a", help="a",type=float,default=1.)
parser.add_argument("-b", help="b",type=float,default=1.)
parser.add_argument("-r","--restart", type=int, nargs="*", help="restart, optional new serial", default=None)
parser.add_argument("-e","--eps", help="epsilon to regularise curvature",type=float,default=0.000001)
parser.add_argument("-IC","--ICfile", help="custom initial condition file",type=str,default=None)
parser.add_argument("-ICx", help="IC erase along x-axis",type=float,default=1.)
parser.add_argument("-ICy", help="IC erase along y-axis",type=float,default=1.)
parser.add_argument("-ICz", help="IC erase along z-axis",type=float,default=1.)
parser.add_argument("--save_cad",type=int,default=2000, help="save_cad")
parser.add_argument("-cpu","--N_threads", help="N_threads",default=None,type=int)
parser.add_argument("-bc","--bcflag", help="boundary conditions mode",type=int,default=1)
parser.add_argument("-f","--freeze", help="impose no evolution on the initial nest portion",type=float,default=None)
parser.add_argument("--bash", help="log on terminal",action="store_true",default=None)

#parser.add_argument("-s","--serial_number", type=int,action="store_true",default=None)

args=parser.parse_args()


#______PARSER FLAGS_____
if args.freeze:
	freeze_flag=1
else:	freeze_flag=0;

#
start_time=datetime.datetime.now()
periodic_flag=1
par_sim=[]


#-----------------create Alias---------------
serial=args.serial
dt=args.dt;		par_sim.append(['TimeStep',dt]);
HD = args.HyperDiffusion; 		par_sim.append(['Hyper-Diffusion',HD]);
AD = args.AntiDiffusion;		par_sim.append(['Anti-Diffusion',AD]);	
Nx=args.Nx;Ny=args.Ny;Nz=args.Nz; par_sim.append(['Nx',Nx]); par_sim.append(['Ny',Ny]); par_sim.append(['Nz',Nz]);	
D =args.Diffusion;			par_sim.append(['Diffusion',D]);
S=args.growth_noise[0];		par_sim.append(['Growth Noise',S])
Lx=args.Lx;Ly=args.Ly;Lz=args.Lz;	par_sim.append(['Lx',Lx]); par_sim.append(['Ly',Ly]); par_sim.append(['Lz',Lz]);
Noise=args.Noise;		par_sim.append(['noise',Noise]);
noise_factor=args.noise_factor;		par_sim.append(['noise_factor',noise_factor]);#This creat an initial random seed of size Lx*2/noise_factor
sim_time=args.sim_time;		par_sim.append(['Max_simultation_time',sim_time]);#This creat an initial random seed of size Lx*2/noise_factor
a=args.a;			par_sim.append(['a',a]);
b=args.b;			par_sim.append(['b',b]);
ADA=args.eps;			par_sim.append(['epsilon',ADA]);
ADx=args.ICx; ADy=args.ICy; ADz=args.ICz;	par_sim.append(['ADx',ADx]); par_sim.append(['ADy',ADy]); par_sim.append(['ADz',ADz]);
save_cad=args.save_cad;		par_sim.append(['save_cadence',save_cad]);      			#saving cadence
periodic_flag=args.bcflag;	par_sim.append(['periodic_flag',periodic_flag]);	

if args.ICfile:
	file_name=args.ICfile	#default restart field
else: file_name="default_restart.hdf5"

N_thr_def=nb.config.NUMBA_DEFAULT_NUM_THREADS 
if not(args.N_threads):
	N_threads=N_thr_def
else: N_threads=args.N_threads




#------Other simulation parameters  (time and save cadence)
iter_slice=1;    	par_sim.append(['iter_slice',iter_slice]);				#number of iter in a single hdf5 file
log_factor=2						#1 over (Nx/64)**3*64/log_factor iterations are registered in the log_file
max_record=300  					#maximum number of recordings
if len(args.growth_noise) > 1:
	nn_sig=args.growth_noise[1]
else: nn_sig=.8							#gaussian smoothening for noise mask
if len(args.growth_noise) > 2:
	noise_cad=int(args.growth_noise[2]/dt)
else: noise_cad=int(1./dt)

print_factor=10;

argv=sys.argv[1:] 		#reading command line input options

#print(argv)

#preparing for data and figure storage
#serial=0

#_________________________determine restart mode___________________________
if not(args.restart==None): 
		print(args.restart)
		file_name=args.serial
		print('restarting from %s' %file_name)
		if len(args.restart)==0:
			serial=int(file_name[file_name.find('_s')-4:file_name.find('_s')])	#serial is the same as restart run
		else: serial = args.restart[0]
		restart_flag=1		
else: restart_flag = 0

#____________________________________create folder to save snapshots____________________________________
save_path='./run_'+ str(serial).rjust(4,'0')
print('save_path is %s'%save_path)
fig_path=save_path + '/snapshots'

if not os.path.exists(save_path):
    os.mkdir(save_path)
else: 
	print("careful this run already exists")
	if restart_flag==0:
		raise NameError("Clean the run folder %s first"%save_path)

if (not os.path.exists(fig_path)) and save_cad<0:
    os.mkdir(fig_path)

#___________________________________master log_________________
mstoutLog = open("./log_master.out", "a", 1)		#open file for master log



#___________________________________particular log_________________
if restart_flag==0:		   
	print(par_sim)
#_______redirecting standard logout
	stdoutLog = open(save_path + "/log_" +str(serial).rjust(4,'0')+".out", "w", 1)
	stdoutLog.write("______________________________________Starting Standard Log________________________________\n")
	stdoutLog.write("---------->START TIME=%s----------\n" %start_time)
	stdoutLog.write("---COMMAND_LINE---\n")
	if not(args.bash):
		sys.stdout = stdoutLog
		sys.stderr = stdoutLog
#	print('This is %s \n'%__file__)	
	print('nohup python3 '+__file__+' '+' '.join(argv[0:]) + ' &>1 &')
	stdoutLog.write("---Simulation parameters---\n")
	print(par_sim)

else:
	stdoutLog = open(save_path + "/log_" +str(serial).rjust(4,'0')+".out", "a", 1)
	stdoutLog.write("__________________Restart from %s  Standard Log__________________________________\n" %file_name)	
	stdoutLog.write("------>RESTART TIME=%s----------\n" %start_time)
	stdoutLog.write("---COMMAND_LINE---\n")
	if not(args.bash):
		sys.stdout = stdoutLog
		sys.stderr = stdoutLog		
	print('nohup python3 '+__file__+''+' '.join(argv[0:]) + ' &>1&')	
#________________________reading in the restart file________________________________
#	[f_h5,x,y,z,iter_0,U_data,ddU_data,eq_pars,eq_flags,grid_pars,save_pars]=h5_restart_3D(file_name)
	[f_h5,x,y,z,iter_0,U_data,eq_pars,eq_flags,grid_pars,save_pars]=h5_restart_3D(file_name) #changed on 18/02/19

#________________________unpacking parameters________________________________
	dt=eq_pars[0]; dx=eq_pars[1]; dy=[2]; dz=eq_pars[3]; HD=eq_pars[4]; AD=eq_pars[5]; D=eq_pars[6] 
	ADA=eq_pars[7]; S=eq_pars[8]; a=eq_pars[9]; b=eq_pars[10]; ADx=eq_pars[11]; ADy=eq_pars[12]; ADz=eq_pars[13]
	periodic_flag=eq_flags[0];laz_flag=eq_flags[1];source_flag=eq_flags[2];jul_flag=eq_flags[3];thin_flag=eq_flags[4];
	save_cad=save_pars[0];iter_slice=save_pars[1]; #max_record=save_pars[2];
	sim_time=args.sim_time; 
	save_cad=args.save_cad;
	dt=eq_pars[0]; 	
	Nx=int(grid_pars[1][0]);
	Ny=int(grid_pars[1][1]);
	Nz=int(grid_pars[1][2]);


#________DETERMINING PRINT MODE AND loGGING rate


T=sim_time #integration time
NT = int(T/dt)  #integration steps



if save_cad<0:			#print pictures or not
#    iter_slice=abs(iter_slice)
    print_flag=1
else:
    print_flag=0		
if  save_cad<0:				#save pictures or not
    save_cad=abs(save_cad)
    save_flag=1
else:
    save_flag=print_flag			
if save_flag*print_flag==0:
#	log_factor=(Nx/64)*64			#log_factor=64 for 64**3 simulation and scale linearly with the number of grid points (in volume)
	log_factor=(max(Nx/64,1))**3*8		#log_factor=64 for 64**3 simulation and scale linearly with the number of grid points (in volume)


save_cad=max(save_cad,NT//max_record)			     	    #save_cadence: when a snapshot is saved
if args.log:
	check_cad=args.log
else:
	check_cad=max(abs(save_cad)//log_factor,1) #save_cad//save_cad      #check cadence: when information are printed on terminal
print_cad=max(abs(save_cad)//print_factor,1)	    		            #print cadence: when figure is updated
#noise_cad=abs(save_cad)
save_pars=[save_cad, iter_slice, max_record,print_factor]
print('save_cadence=%s this simulation will produce %i snapshots' %(save_cad,NT//save_cad))
print('log_cadence=%s this log will contain %i lines' %(check_cad,NT//check_cad))	#found an error 11/06/2019 was '//log_factor' before






#----------Print relevant parameter in a txt file
if restart_flag==0:
	if not os.path.exists(save_path):
	    os.mkdir(save_path)	
	fid=open(save_path + '/par.txt','w+')
	for j in range (0,len(par_sim)):
	    fid.write('%s %f \n'%(par_sim[j][0],par_sim[j][1]))	
	fid.close()

	fid=open('par'+ str(serial).rjust(4,'0') +'.txt','w+')
	for j in range (0,len(par_sim)):
	    fid.write('%s %f \n'%(par_sim[j][0],par_sim[j][1]))
	fid.close()
#--------------copy the command line in a sh file--------------------------
	f_term = open(save_path + "/term_" +str(serial).rjust(4,'0')+".sh", "w", 1)
	f_term.write("python3 " +' '.join(sys.argv[0:]))
	f_term.close()


#READ EQUATION MODE

def read_flags(AD,HD,S,a,Noise,noise_factor):
	#decide of source mode
	if S<0:
	    source_flag=1	#source term is not constant but standard_normal distribution / or coefficient for gradient term
	else:
	    source_flag=0
    #decide of equation type 
	if HD<0:
	    laz_flag=1		#switch to Lazarescu-bis/ter hyper diffusion is without saturation
	else:
	    laz_flag=0
	#decide of saturation exponent 
	if a<0:
	    jul_flag=1		#saturation exponent of source term and anti-diffusion are not the same /gradient term is activated
	else:
	    jul_flag=0

	if AD<0:
	    thin_flag=1		#switch from using laplacian to using curvature in antidiffusion
	else:
	    thin_flag=0

	if Noise<0:		#switch from noisy to not noisy
	    noise_flag=1
	else:
	    noise_flag=0		
	if noise_factor<0:	#switch from rectangular seed to gaussian seed
	    ini_flag=1
	else:
	    ini_flag=0		
	return [laz_flag,source_flag,jul_flag,thin_flag,noise_flag,ini_flag]

laz_flag,source_flag,jul_flag,thin_flag,noise_flag,ini_flag=read_flags(AD,HD,S,a,Noise,noise_factor)
eq_label='div(n)'*thin_flag+'DF'*(1-thin_flag)		#label the growth term, laplacian or div(n)

#___________________________________****INITIAL CONDITIONS****______________________________________
#___________________________________________________________________________________________________


if (restart_flag==0):
	#gridstep
	dx = Lx/(Nx-1) #grid step (space)
	dy = Ly/(Ny-1) #grid step (space)
	dz = Lz/(Nz-1) #grid step (space)
	#group parameters in subgroup
#	save_pars=[]

#-----build parameters lists----------------
	eq_pars=[dt, dx, dy, dz, abs(HD), abs(AD), D, abs(ADA), S, abs(a), b, ADx, ADy, ADz]
#	save_pars=[save_cad, iter_slice, max_record,print_factor]
	grid_pars=[[Lx,Ly,Lz],[Nx,Ny,Nz],abs(Noise),abs(noise_factor)]
	eq_flags=[periodic_flag,laz_flag,source_flag,jul_flag,thin_flag,noise_flag,ini_flag,restart_flag]
	# Grid construction
	x=linspace(0,Lx,Nx-2)
	y=linspace(0,Ly,Ny-2)
	z=linspace(0,Lz,Nz-2)
	#xxx,yyy,zzz=np.meshgrid(x,y,z)    
	xxx,zzz,yyy=np.meshgrid(x,z,y)    
if (restart_flag==0 and not(Noise==0 and noise_factor==0)):
	# Initialisation
	Noise=abs(Noise)
	noise_factor=abs(noise_factor)
	U_data =np.zeros((Nz,Nx,Ny))
	deltax=max(Nx//int(noise_factor/ADx),1); LLx=deltax*dx
	deltay=max(Ny//int(noise_factor/ADy),1); LLy=deltay*dy
	deltaz=max(Nz//int(noise_factor/ADz),1); LLz=deltaz*dz
	noise_ground=0.
	dd=int(noise_factor//Nx);
	print(eq_flags)
	if ini_flag==0:
		U_data[Nz//2-deltaz-dd:Nz//2+deltaz,Nx//2-deltax-dd:Nx//2+deltax,Ny//2-deltay-dd:Ny//2+deltay] = (
		rand(2*deltaz+dd,2*deltax+dd,2*deltay+dd)*Noise*(1-noise_flag)+ones((2 * deltaz+dd,2 * deltax+dd,2 * deltay+dd)) * Noise * noise_flag)
	else:
		U_data[0:Nz-2,0:Nx-2,0:Ny-2]=Noise*((rand(Nz-2,Nx-2,Ny-2)*(1-noise_flag)+noise_flag)*
		np.exp(-(xxx-Lx/2)**2/LLx**2-(yyy-Ly/2)**2/LLy**2-(zzz-Lz/2)**2/LLz**2))
	deltax=deltax//2;deltay=deltay//2;deltaz=deltaz//2			#to make initial square with a hole	
	U_data=U_data+noise_ground		#shifting the base value of the field to non zero 25/01/2018
	if noise_flag==4:
		print('smoothening')
		for i in range (1,100):
			U_data=Timestep_smooth(U_data,dt,dx,dy,dz,HD,AD,D,ADA,S,b,ADx,ADy,ADz,periodic_flag,laz_flag,source_flag,jul_flag,thin_flag,Nx,Ny,Nz)		#let laplacian and bilaplacian smooth initial conditions
			U_data=U_data*abs(Noise)/U_data.max()			
			sys.stdout.write("\r%d%%" % i)
			sys.stdout.flush()	
		print('\n end smoothening')


	if noise_flag==0:
		old_max=U_data.max()
		old_mean=U_data.mean()		
		sig=.5*(1-ini_flag) 						#Large sigma means smoother profiles
		print('Noise_Mask: Gaussian smoothening sigma=%1.2f' %(sig))
		U_data=scfilt.gaussian_filter(U_data-U_data.mean(),sig)		#Apply gaussian filter on the centere
		U_data=U_data*(old_max-old_mean)/U_data.max()+old_mean			#Normalize to the initial max/min values 0)								#Forbid negative values
		U_data=U_data*(old_max)/U_data.max()					 
#		U_data[Nz-2:Nz,0:Nx-2:Nx,Ny-2:Ny]=0				#To avoid divergence at the very first step

#________________reading from default_restart.hdf5 for example we want to start with arbitrary IC and arbitrary parameters____________________
if (restart_flag==0 and args.ICfile):
	if periodic_flag>100 and periodic_flag<10000: #to move copied image to the bottom when imposing dirichlet copied BC
		xb_index=get_digit(periodic_flag,1);xt_index=get_digit(periodic_flag,2);
		if xt_index>2: 
			resize_index=[Nx,Ny,Nz,1]
		elif xb_index>2:
			resize_index=[Nx,Ny,Nz,0]
		else:
			resize_index=[Nx,Ny,Nz]			
	else:
		resize_index=[Nx,Ny,Nz]			
	U_data,_,restart_name=h5_resize_3D(file_name,resize_index)
	print('________Reading from %s_________'%restart_name)	
	#___________________________________****GAUSSIAN BUMP****______________________________________
	#Isosurface are displaced orthogonally to the y-axis (i.e. the last axis of the stack) according to a template which has dimensions Nz*Nx 
	#(i.e. the first two dimensions of the stack) 
#___________________________insert GAUSSIAN BUMP________________________________________
	if (ADx==0. and ADz!=0.):
		print('Gaussian bump %1.2f high and %1.2f wide'%(ADy,ADz))
		x_px=np.linspace(0,Nx-1,Nx)
		z_px=np.linspace(0,Nz-1,Nz)
		xx_px,zz_px=np.meshgrid(x_px,z_px)
		disp=ADy*np.exp(-(xx_px-Nx//2)**2/ADz**2/2-(zz_px-Nz//2)**2/ADz**2/2) #Normal gaussian mask for Ady=1.
		U_data=disp_via_mask(U_data,disp)
#___________________________shift restart image________________________________________
	if (ADz==0.):
		x_IC_disp=np.int(ADx);y_IC_disp=np.int(ADy);
		if x_IC_disp>=0:
			print('Shaving %i px from bottom of x-axis '%x_IC_disp)
			U_data[:,:x_IC_disp,:]=0
		else:
			print('Shaving %i px from top of x-axis '%x_IC_disp)
			U_data[:,x_IC_disp:,:]=0
		if y_IC_disp>=0:
			print('Shaving %i px from bottom of y-axis '%y_IC_disp)
			U_data[:,:,:y_IC_disp]=0
		else:
			print('Shaving %i px from top of y-axis '%y_IC_disp)
			U_data[:,:,y_IC_disp:]=0
	old_max=U_data.max()
	old_mean=U_data.mean()		
	sig=0.
	print('Gaussian smoothening sigma=%1.2f' %(sig))
	U_data=scfilt.gaussian_filter(U_data-U_data.mean(),sig)			#Apply gaussian filter on the center
	U_data=U_data*(old_max-old_mean)/U_data.max()+old_mean			#Normalize to the initial max/min values 
	U_data=U_data*(U_data>0)								#Forbid negative values
	U_data=U_data*(old_max)/U_data.max()					 	



#_____________________determine wall iso_surface____________________
c0=a/(a+b)



#______________Initiating counter__________________
n=0
U_data_back=U_data		#initialize backup field to compute evolution

#create and log the noise mask (to disturb the growing interface)
nn=np.ones((Nz,Nx,Ny))+scfilt.gaussian_filter(np.random.rand(Nz,Nx,Ny)*2*S-S,nn_sig)	
if freeze_flag:
	freeze_mask=(U_data_back<=args.freeze)*1.0	
	nn=freeze_mask*nn	#set to 0 all possible variations inside the pre-existent wall	
	print("%i pre-existent points beyond %1.2f are freezed"%((U_data_back>args.freeze).sum(),args.freeze))
print('created noise mask, S=%1.3f gaussian sigma=%1.2f max=%1.3f min=%1.3f'%(S,nn_sig,nn.max(),nn.min()))
print('noise mask will be updated each %i steps'%noise_cad)

print('print_flag==%i'%print_flag)
U_blank,dU=Timestep(U_data,nn,eq_pars,eq_flags)		 


if restart_flag==0:
	sl_index=0
	sl_name='run_'+ str(serial).rjust(4,'0') + '_s' + str(sl_index) +'.hdf5'
#---------store the very first h5 file as s_0 
	[f_h5, p_dset, dp_dset, iter_dset, nn_dset]=h5_slice_3D(sl_name,x,y,z,U_data,dU,eq_pars,eq_flags,grid_pars,save_pars,save_path)
	iter_dset[()]=0
	iter_0=0
else:
	sl_index=restart_flag*(int(file_name[file_name.find('_s')+2:file_name.find('.hdf5')]))	#sl_index start from the restart slice when this is given
#---------store the very first h5 file as s_res 
	sl_name='run_'+ str(serial).rjust(4,'0') + '_sres.hdf5'
	[f_h5, p_dset, dp_dset, iter_dset, nn_dset]=h5_slice_3D(sl_name,x,y,z,U_data,dU,eq_pars,eq_flags,grid_pars,save_pars,save_path)

#-----Prepare figures
#-----Prepare figures
if print_flag==1:
	ion();  # enables interactive mode
	plotlabel= "t = " + str(n*dt) + '_' + eq_label
	fig,ax=subplots(figsize=(7,9),nrows=1,ncols=1)
	plt00=ax.pcolormesh(U_data[Nz//2,0:-2,0:-2])
	cb00=colorbar(plt00,ax=ax)	
	ax.set_xlabel("x")
	ax.set_xlabel("y")
	ax.set_title("F " )       


save_index=0
curr_time=datetime.datetime.now()
start_time_loop=curr_time


#__________settinge number of core numba should use____________________________
nb.config.NUMBA_NUM_THREADS=N_threads
print('Numba is compiling for %i cpus' %nb.config.NUMBA_NUM_THREADS)


for n in range(0,NT):
	#create a new hdf5 file every save_cad*iterslice iterations
    if (n%(iter_slice*save_cad)==0):
        sl_index=sl_index+1
        f_h5.close()
        save_index=0
        sl_name='run_'+ str(serial).rjust(4,'0') + '_s' + str(sl_index).rjust(4,'0') +'.hdf5'		#11/06/19 I change digits of slice number from 3 to 4
        [f_h5, p_dset,dp_dset, iter_dset, nn_dset]=h5_slice_3D(sl_name,x,y,z,U_data,dU,eq_pars,eq_flags,grid_pars,save_pars,save_path)	
        print("initiating new slice")
   	#print informations every check_cad iterations
#___________________________logging____________________________
    if (n%check_cad==0):
        px_ch=np.abs(U_data-U_data_back).sum()/(Nx*Ny*Nz)	#Here I compute the progressive change in pixel value 
        live_px=(np.abs(U_data-U_data_back)>px_ch).sum()  	#total number of pixel which has changed value
        px_ch_rate=px_ch*100./dt/check_cad
        print("run_%d t=%1.2f/%1.0f slice_%d iter%d/%d-%1.2f%% snap_%d/%d" %(int(serial),(n+iter_0)*dt,(NT+iter_0)*dt,sl_index,n,NT,n/NT*100.,save_index,iter_slice))
        it_time=(datetime.datetime.now()-curr_time).total_seconds()/check_cad		#compute seconds per iteration
        curr_time=datetime.datetime.now()
        el_time=(datetime.datetime.now()-start_time_loop).total_seconds()/3600	#elapsed time in hrs
        print("ps=%i time=%s elap=%1.0f/%1.0fhrs it/sec=%1.4f %s/%s cpus" %(os.getpid(),curr_time,el_time,el_time*NT/(n+1),it_time,N_threads,N_thr_def)) #datetime.datetime.time(
        wall_fraction=(U_data>c0).sum()/(Nx*Ny*Nz)				#wall_fraction
        print("max=%1.6f min=%1.6f wall_fraction=%1.2f%% live_px=%1.2f%% chXpxXsec=%1.5f" %(U_data.max(),U_data.min(),wall_fraction*100.,live_px/(Nx*Ny*Nz)*100.,px_ch_rate))
        U_data_back=U_data			#store field to compare with next slice    
#___________________________printing____________________________ 
    if (n%print_cad==0) and (print_flag==1):        
    	plotlabel="r" + str(serial).rjust(4,'0') + "_t = " + ("%1.3f"%((n+iter_0)*dt)) + " it=" + str(n) + '_' + eq_label
    	fig.suptitle(plotlabel)
    	cb00.remove()		#remove colorbar from previous graph to update both cmap and colorbar
    	plt00=ax.pcolormesh(U_data[Nz//2,0:-2,0:-2])
    	cb00=colorbar(plt00,ax=ax)	
 #___________________________saving____________________________
    if (n%save_cad==0):
                if save_flag==1: 
	                fig.savefig(fig_path + '/FD_' + 'r'+ str(serial).rjust(4,'0') + '_iter_' + str(int(n)).rjust(len(str(int(NT))),'0') + '.png') 
                iter_dset[()]=n+iter_0		#warning even for single value data, dataset id must be followed by [()] to set its value
                p_dset[:,:,:]=U_data		#store current field
                dp_dset[:,:,:]=dU			#store current increment
                nn_dset[:,:]=nn[Nz//2,:,:]	#store current noise mask

#                dd_dset[save_index,:,:,:]=dUdU_data    #remove from version sub_8
                mstoutLog.write("__new_slice__run=%i ps=%i t=%1.2f/%1.0f elap=%1.0f/%1.0fhrs slice_%i/%i %s/%scpus live_px=%1.2f%% chXpxXsec=%1.5f \n" %(int(serial),os.getpid(),(n+iter_0)*dt,(NT+iter_0)*dt,el_time,el_time*NT/(n+1),sl_index,NT//save_cad,N_threads,N_thr_def,live_px/(Nx*Ny*Nz)*100.,px_ch_rate))
    if (n%noise_cad==0):
				#update the noise mask				
                nn=np.ones((Nz,Nx,Ny))+scfilt.gaussian_filter(np.random.rand(Nz,Nx,Ny)*2*S-S,nn_sig)
                print("update noise mask S=%1.2f gauss_sigma=%1.2f max=%1.3f min=%1.3f"%(S,nn_sig,nn.max(),nn.min()))
                if freeze_flag:
	                nn=nn*freeze_mask	#set to 0 all possible variations inside the pre-existent wall	                print("update noise mask S=%1.2f gauss_sigma=%1.2f max=%1.3f min=%1.3f"%(S,nn_sig,nn.max(),nn.min()))

	#update the noise mask
#    nn=np.ones((Nz,Nx,Ny))+scfilt.gaussian_filter(np.random.rand(Nz,Nx,Ny)*2*S-S,nn_sig)	
	#main time step
    U_data,dU= Timestep(U_data,nn,eq_pars,eq_flags)
mstoutLog.write("___________________END PROCESS___run=%i ps=%i t=%1.0f time=%s cpu_time=%1.0fhrs slice_%i/%i %scpus live_px=%1.2f%% chXpxXsec=%1.5f \n" %(int(serial),(n+iter_0)*dt,os.getpid(),curr_time,el_time/N_threads,sl_index-1,NT//save_cad,N_threads,live_px/(Nx*Ny*Nz)*100.,px_ch_rate))
f_h5.close()           #close hdf5 file




