# Makefile to remove specific files from the current directory

# Define the files to be removed
FILES_TO_REMOVE := $(wildcard *.fits) amr_grid.inp stars.inp cost.out dust_density.inp dust_temperature.dat dustopac.inp image.out wavelength_micron.inp radmc3d.inp radmc3d.out

# Define the target to remove files
clean:
	@echo "Removing files..."
	@rm -f $(FILES_TO_REMOVE)
	@echo "Files removed successfully."