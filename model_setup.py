import astropy.constants as const
import astropy.units as u
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import time
import matplotlib.pylab as plb
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from spectral_cube import SpectralCube
from spectral_cube import Projection
import radio_beam
from radio_beam import Beam
from astropy.io import fits
from radmc3dPy.image import *    
from radmc3dPy.analyze import *  
from radmc3dPy.natconst import * 
from astropy.wcs import WCS
import sys
#sys.path.append('../../SED_sgrb2/SED_fit')
#from fit import extract_dimensions

'''3D model with layered AMR'''

class model_setup_3Dspher:
    def __init__(self,rho0,prho,nphot=100000,laynr=2):
        self.laynr = laynr

        #
        # Star parameters
        #
        self.mstar    = 30*const.M_sun # in kg, check if correct units
        self.rstar    = 13.4*const.R_sun # in m , check units
        self.tstar    = 30000*u.K #in K 
        self.pstar    = np.array([0.,0.,0.]) #position in cartesian coords
        #
        # Wavelengths - this eventually needs a function to calculate it based of start and endpoint and maybe number of intervals.
        #
        lam1     = 0.1e0
        lam2     = 7.0e0
        lam3     = 25.e0
        lam4     = 1.0e4
        lam5	 = 9.0e4
        n12      = 100
        n23      = 100
        n34      = 100
        n45	     = 100
        lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
        lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
        lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=False)
        lam45    = np.logspace(np.log10(lam4),np.log10(lam5),n45,endpoint=True)
        self.lam      = np.concatenate([lam12,lam23,lam34,lam45])
        self.nlam     = self.lam.size

        # grid parameters - eventually replace this with a function or put as argument.
        self.nx = 100  # 
        self.ny = 100  # 
        self.nz = 100  # 
        self.laynlev = 2  # I THINK THIS IS HOW MUCH FINER IT IS

        # Initialize additional parameters based on laynr
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize parameters based on the number of layers.
        xi_ls    = []
        yi_ls    = []
        zi_ls    = []
        xc_ls    = []
        yc_ls    = []
        zc_ls    = []
        for i in range(self.laynr):
            xi_l=np.linspace(i*-self.sizex/self.laynr,i*self.sizex/self.laynr,i*self.lnnx+1)
            yi_l=np.linspace(i*-self.sizey/self.laynr,i*self.sizey/self.laynr,i*self.lnny+1)
            zi_l=np.linspace(i*-self.sizez/self.laynr,i*self.sizez/self.laynr,i*self.lnnz+1)
            xi_ls.append(xi_l)
            yi_ls.append(yi_l)
            zi_ls.append(zi_l)
            xc_ls.append(0.5 * (xi_l[0:self.nx]+xi_l[1:self.nx+1]))
            yc_ls.append(0.5 * (yi_l[0:self.yx]+yi_l[1:self.yx+1]))
            zc_ls.append(0.5 * (zi_l[0:self.zx]+zi_l[1:self.zx+1]))




    def write_input(self,amr=False,mrw=False):
        # Write the wavelength file
    
        with open('wavelength_micron.inp','w+') as f:
            f.write('%d\n'%(self.nlam))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
        #
        #
        # Write the stars.inp file
        #
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(self.rstar,self.mstar,self.pstar[0],self.pstar[1],self.pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
            f.write('\n%13.6e\n'%(-self.tstar))
        #
        # Write the grid file
        #
        with open('amr_grid.inp','w+') as f:
            f.write('1\n')                       # iformat
            f.write('10\n')                      # AMR grid style  (10=layer-style AMR)
            f.write('0\n')                       # Coordinate system
            f.write('0\n')                       # gridinfo
            f.write('1 1 1\n')                   # Include x,y,z coordinate
            f.write('%d %d %d\n'%(self.nx,self.ny,self.nz))     # Size of grid
            f.write('%d %d\n'%(self.laynlev,self.laynr))   # Layers: nr of levels, nr of layers
            for value in self.xi:
                f.write('%13.6e\n'%(value))      # X coordinates (cell walls)
            for value in self.yi:
                f.write('%13.6e\n'%(value))      # Y coordinates (cell walls)
            for value in self.zi:
                f.write('%13.6e\n'%(value))      # Z coordinates (cell walls)
            f.write('%d %d %d %d %d %d %d\n'%(0,self.lix,self.liy,self.liz,self.lnx,self.lny,self.lnz))     # Info for layer


        #
        # Write the density file
        #
        with open('dust_density.inp','w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n'%(self.nx*self.ny*self.nz+self.lnnx*self.lnny*self.lnnz)) # Nr of spatial data points (incl redundant ones)
            f.write('1\n')                       # Nr of dust species
            data = self.rhod.ravel(order='F')         # Create a 1-D view of rhod
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
            data = self.rhod_l1.ravel(order='F')      # Create a 1-D view of rhod_l1
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

        #
        # Dust opacity control file
        #
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('1               Nr of dust species\n')
            f.write('============================================================================\n')
            f.write('1               Way in which this dust species is read\n')
            f.write('0               0=Thermal grain\n')
            f.write('silicate        Extension of name of dustkappa_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
        #
        # Write the radmc3d.inp control file
        #
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(self.nphot))
            f.write('scattering_mode_max = 0\n')   # Put this to 1 for isotropic scattering
            f.write('iranfreqmode = 1\n')
            if mrw:
                f.write('modified_random_walk = 1')

    def calculate_model(self):
        t0 = time.time()
        os.system('radmc3d mctherm')
        #os.system('radmc3d sed')
        t1 = time.time()

        total = t1-t0
        print("Calculating the model cost: "+str(total)+" s")
        with open('cost.out','w+') as f:
            f.write("Calculating the model cost: "+str(total)+" s\n")
        #Make the necessary calls to run radmc3d
        
    def sed(self):
        #plot sed
        s    = readSpectrum()
        plt.figure()
        lammic = s[:,0]
        flux   = s[:,1]
        nu     = 1e4*const.c.cgs/lammic
        nufnu  = nu*flux
        nulnu  = nufnu*4*math.pi*(const.pc.cgs)*(const.pc.cgs)
        plt.plot(lammic,nulnu/self.ls)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda$ [$\mu$m]')
        plt.ylabel(r'$\nu L_{\nu}$ [$L_{\odot}$]')
        plt.axis()
        plt.show()
    
    def make_synth_maps(self,wls):
        t0 = time.time()
        for wl in wls:
            os.system('radmc3d image lambda '+str(wl))
            im   = readImage()
            im.writeFits('model'+str(wl)+'.fits')
            process_radmc_image('model'+str(wl)+'.fits','model'+str(wl)+'_smooth.fits',0.4,overwrite=True)
            cim = im.imConv(fwhm=[0.4, 0.4], pa=0., dpc=8340.) 
            #plt.figure()
            #plotImage(cim, arcsec=True, dpc=8340., cmap=plb.cm.gist_heat)
            #cim.writeFits('model'+str(wl)+'.fits', dpc=8340.)
            #plt.show()
        t1 = time.time()
        total=t1-t0
        print("Calculating the images cost: "+str(total)+" s")
        #plot synthetic maps at each wavelength specified.

    def make_tau_surface(self,wls):
        for wl in wls:
            os.system("radmc3d tausurf 1.0 lambda "+str(wl))
            im   = readImage()
            data = np.squeeze(im.image[:, ::-1, 0].T)
            plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('tau_surf_'+str(wl)+'.fits')
            wcs = WCS(fits.getheader('tau_surf_'+str(wl)+'.fits'))
            newhdu = fits.PrimaryHDU(data,wcs.to_header())
            newhdu.writeto('tau_surf_'+str(wl)+'.fits',overwrite=True)

    def make_tau_map(self,wls):
        for wl in wls:
            os.system("radmc3d image lambda "+str(wl)+" tracetau")
            im   = readImage()
            data = np.squeeze(im.image[:, ::-1, 0].T)
            #plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('tau'+str(wl)+'.fits')
            wcs = WCS(fits.getheader('tau'+str(wl)+'.fits')).celestial
            newhdu = fits.PrimaryHDU(data,wcs.to_header())
            newhdu.writeto('tau'+str(wl)+'.fits',overwrite=True)

    def make_column_density_map(self,wls):
        for wl in wls:
            os.system("radmc3d image lambda "+str(wl)+" tracecolumn")
            im   = readImage()
            data = np.squeeze(im.image[:, ::-1, 0].T)
            #plotImage(im,arcsec=True,dpc=8340.)
            im.writeFits('column_density_'+str(wl)+'.fits')
            #open fits file, get wcs, overwrite
            wcs = WCS(fits.getheader('column_density_'+str(wl)+'.fits')).celestial
            newhdu = fits.PrimaryHDU(data,wcs.to_header())
            newhdu.writeto('column_density_'+str(wl)+'.fits',overwrite=True)

    
    def density_profile(self):
        #plot density vs radius
        au=const.au.cgs.value # AU [cm]
        a    = readData(ddens=True,binary=False)
        r    = a.grid.x[:]
        density = a.rhodust[:,0,0,0]
        plt.figure(1)
        plt.plot(r/au,density)
        plt.xlabel('r [au]')
        plt.ylabel(r'$\rho_{dust}$ [$g/cm^3$]')
        plt.show()
        plt.savefig('density.png')

