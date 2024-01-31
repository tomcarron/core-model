from model_setup import model_setup_3Dspher as model
import astropy.constants as const

rho0=1e5 *1.6735575e-24   #reference density cgs units
prho=-1.0 #power law index
test=model(rho0,prho,size=5000,nphot=100000,radius=(3000*const.au).to("cm").value)

#write input file
test.write_input(mrw=True)

#run model
test.calculate_model()

wls=[450] #micron
test.make_synth_maps(wls)