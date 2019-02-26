import rebound 
import numpy as np
import matplotlib.pyplot as plt

##Get com for a list of rebound particles.
def get_com(ps):
	'''
	Get center of mass for a collection of particles
	'''
	com=ps[0].m*ps[0]
	ms=ps[0].m
	for pp in ps[1:]:
		com=com+pp.m*pp
		ms+=pp.m
	return com/ms


def bin_props(p1, p2):
	'''
	Auxiliary function to get binary properties for two particles. 

	p1 and p2 -- Two particles from a rebound simulation.
	'''
	dp = p1 - p2   
	d2 = dp.x*dp.x+dp.y*dp.y+dp.z*dp.z
	##Calculate com velocity of two particles...
	##Masses
	m1=p1.m
	m2=p2.m
	com = (m1*p1+m2*p2)/(m1+m2)
	##Particle velocities in com frame
	p1_com = p1 - com
	p2_com = p2 - com
	v12 = (p1_com.vx**2.+p1_com.vy**2.+p1_com.vz**2.)
	v22 = (p2_com.vx**2.+p2_com.vy**2.+p2_com.vz**2.)

	d12 = (p1_com.x**2.+p1_com.y**2.+p1_com.z**2.)
	d22 = (p2_com.vx**2.+p2_com.y**2.+p2_com.z**2.)	
	##Difference in the forces acting on the two particles;
	ft = np.array([m2*(p2.ax)-m2*(com.ax), m2*(p2.ay)-m2*(com.ay), m2*(p2.az)-m2*com.az])
	##Unit vector pointing from particle 2 to particle 1
	rhat = np.array(dp.xyz)/d2**0.5
	f12 = m1*m2/d2*rhat 
	##Tidal force that star 2 experiences
	ft = ft - f12
	ft = np.linalg.norm(ft)

	##Kinetic and potential energies
	ke = 0.5*m1*v12+0.5*m2*v22
	##Potential energy; Assumes G = 1
	pe = (m1*m2)/d2**0.5

	##Distance of binary center of mass from COM of system (should be near central SMBH)
	com_d=(com.x**2.+com.y**2.+com.z**2.)**0.5
	a_bin=(m1*m2)/(2.*(pe-ke))
	##Angular momentum in binary com
	j_bin=m1*np.cross(p1_com.xyz, p1_com.vxyz)+m2*np.cross(p2_com.xyz, p2_com.vxyz)
	##Angular momentum of binary com
	j_com=(m1+m2)*np.cross(com.xyz, com.vxyz)

	#Inclination of star's orbit wrt the binary's orbit win the disk 
	inc=np.arccos(np.dot(j_bin, j_com)/np.linalg.norm(j_bin)/np.linalg.norm(j_com))*180./np.pi
	mu=m1*m2/(m1+m2)
	##Eccentricity of the binary
	e_bin=(1.-np.linalg.norm(j_bin)**2./((m1+m2)*a_bin)/(mu**2.))

	return com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft



def com_plot(sa_name, i1, i2, extras=[], name='', cols=['r', 'b'], cols_extra=['k'], idx_min=0, idx_max=None, lim=0.1):
	'''
	Gets positions of particles i1 and i2 for every snapshot in SimulationArchive sa_name 
	and plots the positions of these particles in the com frame. It plots the projections of the particles
	in the xy, xz, and yz planes in three separate frames.

	By default saves all of the snapshots in the archive to a separate png
	file (named sa_name_zzz.png, where zzz indicate the index of the snapshot). The name kwarg can be used to add an extra
	tag between sa_name and zzz. With idx_min and idx_max the user can specify the minimum and maximum snapshot index
	to include.

	The function calls the auxiliary function function bin_props (defined above) 
	to extract the positions of the particles as well as the binary orbital properties of particles i1 and i2. The 
	binary orbital parameters added as an annotation to the middle panel 

	The user may pass extra particles to plot via the extra kwarg (this is useful in case there is an 
	exchange with the binary). The colors of the points are controlled by cols and cols_extra.
	The axis ranges are controlled by lim.

	'''

	##Setting up plots: axis labels and ranges
	planes = [['x', 'y'], ['x', 'z'], ['y','z']]
	fig,ax=plt.subplots(nrows=1, ncols=3, figsize=(10*len(planes),10))
	for kk in range(3):
		ax[kk].set_xlim(-lim,  lim)
		ax[kk].set_ylim(-lim, lim)
		ax[kk].set_xlabel(planes[kk][0])
		ax[kk].set_ylabel(planes[kk][1])

	#Load sim archive
	sa = rebound.SimulationArchive(sa_name)
	if not idx_max:
		idx_max=len(sa)
	m0=sa[0].particles[0].m
	##Iterate over snapshots and projections.
	for ii in range(idx_min, idx_max):
		for kk, plane in enumerate(planes):
			sim=sa[ii]
			p1,p2=sim.particles[i1],sim.particles[i2]
			m1,m2=p1.m,p2.m

			##Getting binary properties
			com_d, a_bin, e_bin, p1_com, p2_com, d2, inc, ft = bin_props(p1,p2)
			p1_pos=getattr(p1_com, plane[0]), getattr(p1_com, plane[1])
			p2_pos=getattr(p2_com, plane[0]), getattr(p2_com, plane[1])
			com=get_com([p1, p2])
			#Plotting positions of binary stars. 
			ax[kk].plot(p1_pos[0], p1_pos[1], 'o', markersize=2, color=cols[0])
			ax[kk].plot(p2_pos[0], p2_pos[1], 'o', markersize=2, color=cols[1])
			for jj, extra in enumerate(extras):
				ax[kk].plot(getattr(sim.particles[extra]-com, plane[0]), getattr(sim.particles[extra]-com, plane[1]), 'o', markersize=2, color=cols_extra[(jj)%len(cols_extra)])
		##Annotate plot middle panel with binary parameters 
		ann=ax[1].annotate('a={0:2.2g}, a,/rt={1:2.2g}, r={2:2.2g}\n e^2={3:2.2g}, 1-e^2={4:2.2g}\n i={5:2.2g}'\
			.format(a_bin, a_bin/(((m1+m2)/m0)**(1./3.)*com_d), com_d, e_bin, 1-e_bin, inc),\
			(0.9*lim, 0.9*lim), horizontalalignment='right',verticalalignment='top', fontsize=20)
		fig.savefig(sa_name.replace('.bin', '')+name+'_com_{0:03d}.png'.format(ii), bbox_inches='tight', pad_inches=0)
		ann.remove()

name='archive_bin1.bin'
com_plot(name, 35, 82, cols=['r', 'b'], name='_cool', lim=0.06)
##pngs can be turned into a movie with the following bash command: ffmpeg -r 3 -i archive_bin1_cool_com_%3d.png demo.mp4