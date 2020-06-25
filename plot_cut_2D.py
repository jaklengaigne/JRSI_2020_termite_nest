#This version adapted from plot_cut_2_paper.py  250620 for JRSI publication




import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
from IPython import display
import numpy.fft as fft
import argparse







class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


data_path='./' 			#where is situated your hdf5 file
save_path = './'		#where you want to save snapshot 

#__________________________________BUILD PARSER__________________________________________
#________________________________________________________________________________________
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="file to analyse")
parser.add_argument("--nosave", help="figures are not saved",
                    action="store_true", default=False)
parser.add_argument("--notitle", help="no title",
                    action="store_true", default=False)
parser.add_argument("-i", "--increment", help="plot increment instead of f-field",
                    action="store_true")
parser.add_argument("--swap", help="swap z-axis with x-1 or y-",
                    type=int)
parser.add_argument("-c","--colorbar", help="add colorbar to single plot",
                    action="store_true",default=False)
parser.add_argument("-t","--timescale", help="normalize time with the given frequency",
                    type=float,default=None)
parser.add_argument("--clim",help="colorbar limit",nargs="*",
                    type=float,default=None)
parser.add_argument("-l", "--levelset", help="assign levelset",type=float,
                    default=0.)
parser.add_argument("-k", "--keep", help="keep previous figures",
                    action="store_true",default=False)
parser.add_argument("-z", "--cut_index", type=int, help="height for the 2D cut in px",
                default=None)
parser.add_argument("-v","--verbose", help="show hdf5 content",
                    action="store_true", default=False)
parser.add_argument("--axis", help="plot axis ticks",
                    action="store_true", default=False)
parser.add_argument("-s", "--shave", help="shave outer pixels",nargs="*",type=int,
                    default=[2])
parser.add_argument("-n", "--noise", help="plot noise-mask instead of f-field",
                    action="store_true")

args=parser.parse_args()


#__________________________________CREATING ALIAS________________________________________
#________________________________________________________________________________________
file_name=args.filename
c0=args.levelset
save_flag=not(args.nosave)
NN=args.shave

if not(args.keep):
	plt.close('all')



f = h5py.File(file_name,'r')  #after update <-----giulio
Nx=f["grid"]["Nx"][()]
Ny=f["grid"]["Ny"][()]
Nz=f["grid"]["Nz"][()]






#I read in the HDF5 file


#I print the HDF5 file field
if args.verbose:
	print ("%s contains:" %file_name)
	for field in f:
		print("->%s"%field)
		if field!="name":
			for s_field in f[field]:
				print("______-->%s"%s_field)


#__________grid shaving___________________________
if len(NN)==1:
	NNx=NNy=NNz=max(NN[0],0)
elif len(NN)==3:
	NNx=max(NN[0],0);NNy=max(NN[1],0);NNz=max(NN[2],0);
else:
	parser.error("Pixel to shave must be a list 1 or 3 elements long ")
print("shaving %i px along x %i px along y %i px along z"%(NNx,NNy,NNz))

if args.cut_index==None:
	z_index=(Nz-2*NNz)//2
else:
	z_index=args.cut_index 




#creating alias to plot
###SCALES########
if "iter" in f["fields"]:
	iteration=f["fields"]["iter"][()]
else: iteration=0
if isinstance(iteration,np.ndarray):
#if len(iteration)>1:
	iteration=iteration[0]
if "p" in f["fields"]:
	ppp=f["fields"]["p"][()]
else: 	ppp=f["fields"]["c_field_bandpass"][()]/256
if len(ppp.shape)==4:	
	ppp=(f["fields"]["p"][-1,:,:,:]).reshape(Nz,Nx,Ny)
if args.increment and "dp" in f['fields']:
	ppp=f["fields"]["dp"][()]


AD=0;

#kx=f["scales"]["kx"][:]
dx=f["grid"]["dx"][()]
dy=f["grid"]["dy"][()]
dz=f["grid"]["dz"][()]
Lx=f["grid"]["Lx"][()]
Ly=f["grid"]["Ly"][()]
Lz=f["grid"]["Lz"][()]
#thin_flag=f["flags"]["thin_flag"][()]
#eps=f["parameters"]["ADA"][()]
if "a" in f["parameters"]:
	eps=0.000001
	AD=f["parameters"]["AntiDiffusion"][()]
	D=f["parameters"]["Diffusion"][()]
	HD=f["parameters"]["HyperDiffusion"][()]
	a=f["parameters"]["a"][()]
	b=f["parameters"]["b"][()]
	dt=f["grid"]["dt"][()]
else: AD=D=HD=dt=0; a=b=1.


x=f["grid"]["x"][:].squeeze()
y=f["grid"]["y"][:].squeeze()
z=f["grid"]["y"][:].squeeze()


N=0	#pixels shaved from upper boundaries

U=ppp[NNz:Nz-NNz,NNx:Nx-NNx,NNy:Ny-NNy];



#_________________swap axis___________
if args.swap:
	U=U.swapaxes(0,args.swap)



p=U[z_index,:,:]   #pick the mid-z value to plot 2D-slices 

#plot noise mask cut if requested
if args.noise and "nn" in f['fields']:
	p=f["fields"]["nn"][()]





L0=Lx/2.		#center of the plot

if c0==0:
	c0=a/(a+b)
	print('c0=a/(a+b)=%1.3f'%c0)
	


T= iteration * dt  #compute the corresponding time	

if args.timescale:
	if args.timescale==1:
		print('rescale time by most unstable mode')
		T *= c0 ** 2 * AD ** 2 / 2 / np.pi / 4
	else:
		T *= args.timescale / 2 / np.pi


#################PLOT###########
ym, xm = np.meshgrid(y[NNy:Ny-NNy],x[NNx:Nx-NNx])
c_field=U	





#------------->prepare figure
plt.ion()
cmap_0=matplotlib.cm.PuOr.reversed()
fontsingle=45
run_name=file_name[file_name.find('_s')-4:file_name.find('.h')]

if not args.clim:
	elev_min=c_field.min()
	elev_max=c_field.max()
	mid_val=c0
else:
	elev_min=float(args.clim[0])
	elev_max=float(args.clim[1])
	mid_val=(elev_max+elev_min)/2




fig2,ax2=plt.subplots(nrows=1,ncols=1,figsize=(5.5,5.5))
plot=ax2.pcolormesh(p,cmap=cmap_0,clim=(elev_min, elev_max),norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))#,rasterized = True)
plot.set_edgecolor('face')
#	ax2.set_title('R%s t=%1.2f'% (run_name[0:run_name.find('_')],T),fontsize=fontsingle)
if not (args.notitle):
	ax2.set_title('t=%1.0f'%T,fontsize=fontsingle)
if not(args.axis):
	ax2.axis('off')
ax2.axis('equal')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set_frame_on(False)
#	plt.xlabel('x',fontsize=fontsingle)
#	plt.ylabel('y',fontsize=fontsingle)
if args.colorbar:
	clb=plt.colorbar(plot,ax=ax2)
	ticklabs = clb.ax.get_yticklabels()
#		clb.ax.set_yticklabels(ticklabs, fontsize=fontsingle)
#	ax2.contour(c_field[z_index,:,:],[c0],linestyles='dashed',colors='k')
if save_flag==1:
	if not args.increment:
		save_suff='_s'
		if args.timescale:
			save_suff += '_norm'
	else: save_suff='_s_inc' 	
	fig2.savefig(save_path + run_name + save_suff, bbox_inches='tight')


plt.show()



