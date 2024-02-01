from model_setup import model_setup_3Dspher as model
import astropy.constants as const

rho0=1e5 *1.6735575e-24   #reference density cgs units
prho=-1.0 #power law index
test=model(rho0,prho,size=5000,nphot=100000,radius=3000)
test.add_gaussian_variations(0.2)

#write input file
test.write_input(mrw=True)

#run model
test.calculate_model(ncores=28)

#sed
#test.sed()

#density profile
test.density_profile()
test.temperature_profile()

wls=[450,850,1000,2000,3000] #micron
#these functions take some time and could benefit from multithreading too.
#test.make_synth_maps(wls)
#test.make_tau_surface(wls)