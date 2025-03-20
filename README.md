# cholla_lya_scripts

Python scripts to study the Lyman-alpha Forest in cosmological Cholla simulations. 

# Motivation

We would like to study the optical depth and transmitted flux power spectrum in a cosmological Cholla simulation from On-The-Fly Skewer files.

We expect the following attributes in each Skewer file.

1. ``Lbox`` - array of 3 floats, detailing length of simulated box in each dimension in units of $\textrm{kpc}$
2. ``Omega_R`` - Present-Day Radiation Energy Density
3. ``Omega_M`` - Present-Day Matter Energy Density
4. ``Omega_L`` - Present-Day Dark Energy Density
5. ``Omega_K`` - Present-Day Spatial Curvature Energy Density
6. ``w0`` and ``wa`` - parameters specifying time-evolving Dark Energy equation of state
7. ``current_a`` and ``current_z`` - scale factor and redshift at which skewer data is taken
8. ``H0`` - Present-Day Hubble parameter in units of $\textrm{kpc} / \textrm{s} / \textrm{Mpc}$

Most scripts have been written with the python package [argparse](https://docs.python.org/3/howto/argparse.html) which allows for a quick command line interface.


# Optical Depth Calculation

We assume a Gaussian line profile. To calculate the optical depth, we run

```bash
$ python3 optdepth.py $SKEWERFILE -v -l
```

where 

1. ``$SKEWERFILE`` is the one positional argument - the skewer output file
2. ``-v`` flags the script to be verbose throughout the calculation
3. ``-l`` flags the script to save the local optical depth

To only save the median optical depth, and not the local optical depth, do not include the ``-l`` flag.

In practice, ``optdepth.py`` scales very poorly for larger simulation boxes, so we seek someway to hasten the calculation. The Python package [mpi4py](https://mpi4py.readthedocs.io) provides Python bindings for the message passing interface (MPI) standard which is common in many parallelized codes, including Cholla itself. Instead of using Python, we could write this calculation in C itself, but one step at a time bro chilllllll. From serially looping over each skewer individually in ``optdepth.py``,

```python
for nSkewerID in range(nSkewers):
    ...
    tau_local[nSkewerID] = tau
```

we instead assign specific skewer IDs for each processor

```python
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

skewerID_arr = np.arange(nSkewers)
skewerIDs_rank = np.arghwere((skewerID_arr % size) == rank).flatten()

for nSkewerID in skewerID_rank:
    ...
    tau_local[nSkewerID] = tau
```

If we have 10 skewers and 4 processors, we have the following ranks responsible for following skewer IDs

1. rank 0 - [0,4,8]
2. rank 1 - [1,5,9]
3. rank 2 - [2,6]
4. rank 3 - [3,7]

This is all great and good and amazing, but there is the issue of actually grabbing the data. To open and write data onto the skewer HDF5 files, we use the Python package [h5py](http://docs.h5py.org) which provides a Pythonic interface for HDF5 format files. Luckily, the developers for h5py have provided detailed instructions on building h5py that utilizes Parallel HDF5 [here](https://docs.h5py.org/en/stable/mpi.html).

Details on how we accomplished this on [lux](https://lux-ucsc.readthedocs.io) is found in the file `create_h5pympi.txt`.

The parallelized script runs with your favorite MPI standard

```bash
$ mpirun -np $NUMNODES python3 optdepth_mpi.py $SKEWERFILE -v
```

where the `-np` flag specifies the number of processors to use in the calculation, specified here with `$NUMNODES`.




# Power Spectrum Binning

We would like to study cosmological simulations that have different expansion histories. A good probe for this will be to individually probe optically thick regions compared against skewers that probe the optically thin regime.

We don't want to arbitrarily choose some optical depth that serves as the barrier between optically thin and thick, so instead we place skewers in bins of effective optical depth, and take the flux power spectrum for each seperately. Unaware of any specific optical depth to take seriously, instead of placing strict limits, we first allow for a range of optical depth and some number of quantiles to split these effectice optical depths into. After calculating the optical depth, we can then bin skewers and take the stacked flux power spectra.

## Quantile Binning

The structure to group skewers by optical depth using some number of quantiles is

```bash
$ python3 powspec_skewer_quantiles.py $SKEWERFILE $NQUANTILE $OPTDEPTHLOW $OPTDEPTHUPP -v
```

As can be deduced from the variables names, the arguments for the python script are

1. `$SKEWERFILE` - skewer output file
2. `$NQUANTILE` - the number of quantiles to use
3. `$OPTDEPTHLOW` - lower effective optical depth range for most optically thin regime
4. `$OPTDEPTHUPP` - upper effective optical depth range for most optically thick regime

The `-v` flag tells the script to be verbose thorughout the calculation. We expect the skewer file to have the format `nOutput_skewers.h5` where `nOutput` is the output number from the Cholla outputs. From this Python script, we produce an HDF5 file `nOutput_fluxpowerspectrum_optdepthbin.h5` placed in the parent directory of the directory the skewer file resides in. This file has the general structure

```bash
nOutput_fluxpowerspectrum_optdepthbin.h5
├── attrs
├── FluxPowerSpectrum_quantile_0
│   └── attrs
│   └── indices
│   └── FPS_x
│   └── FPS_y
│   └── FPS_z
├── FluxPowerSpectrum_quantile_1
│   └── attrs
│   └── indices
│   └── FPS_x
│   └── FPS_y
│   └── FPS_z
...
├── FluxPowerSpectrum_quantile_NQUANTILE
│   └── attrs
│   └── indices
│   └── FPS_x
│   └── FPS_y
│   └── FPS_z
└── ...
```

The attributes attached to the root group `nOutput_fluxpowerspectrum_optdepthbin.h5` include global information about the simulation, output information of the skewer file, and analysis information. The following values are attached in this attribute

1. ``Omega_R`` - Present-Day Radiation Energy Density
2. ``Omega_M`` - Present-Day Matter Energy Density
3. ``Omega_L`` - Present-Day Dark Energy Density
4. ``Omega_K`` - Present-Day Spatial Curvature Energy Density
5. ``Omega_b`` - Presend-Day Baryonic Matter Energy Density
5. ``w0`` and ``wa`` - parameters specifying time-evolving Dark Energy equation of state
6. ``H0`` - Present-Day Hubble parameter in units of $\textrm{kpc} / \textrm{s} / \textrm{Mpc}$
7. ``current_a`` and ``current_z`` - scale factor and redshift at which skewer data is taken
8. ``fPath`` - file path to the skewer file
9. ``nOutput`` - the index output value with respect to the rest of the simulation
10. ``Lbox`` - array of 3 floats, detailing length of simulated box in each dimension in units of $\textrm{kpc}$
11. ``nCells`` - array of 3 ints, detailing number of cells used in simulated box in each dimension
12. ``nStrides`` - array of 3 ints, detailing the number of cell-strides taken between skewers in each dimension
13. ``nSkewers`` - array of 3 ints, detailing the number of skewers in each dimension
14. ``tau_eff_low`` and ``tau_eff_upp`` - input arguments detailing the upper and lower effective optical depth to use in tiling
15. ``tau_eff_mean`` - mean of all effective optical depth
16. ``nquantiles`` - input argument detailing the number of quantiels to use in tiling

The datasets attached to the root group are the following

1. ``k_x``, ``k_y``, and ``k_z`` - arrays of size `1. + (nCells / 2.)` that hold the k-mode values in each dimension in units of $\textrm{s}\ \textrm{km}^{-1}$


The attributes attached to each quantile group ``FluxPowerSpectrum_quantile_0`` give details on this quantile region. The following values are attached in this attribute

1. ``tau_min`` and ``tau_max`` - minimum and maximum effective optical depth defining this quantile
2. ``tau_mean`` - mean of the effective optical depths landing in this quantile

The datasets inside each quantile group are mostly straigthforward

1. ``indices`` - the indices of the skewers that land in this quantile. 
2. ``FPS_X``, ``FPS_Y``, and ``FPS_Z`` - the flux power spectrum in each dimension.

What is the `indices` dataset in detail? Well in organizing the skewers, we use its index in the local optical depth dataset (saved in the shape of (`nSkewers`, `nLOS`) for some axis) as an ID for this skewer. This array holds those ID values. If we take all skewers independent of axis when sorting, then how do we distinguish between axis in ``indices``? These values will run from zero to the total number of skewers summed along all axes, so we just choose to have indices `(0, nCells[0])` correspond to skewer IDs along the x-axis, `(nCells[0], nCells[1])` to skewer IDs along the y-axis, and `(nCells[0] + nCells[1], )` to skewer IDs along the z-axis.

We have as many ``FluxPowerSpectrum_quantile`` groups as specified by the input `$NQUANTILE` argument into `powspec_skewer_quantiles.py`.


## Grouping Flux Power Spectra

Beautiful! Well, how do we know whether this actually works? In order to do this, we need some way to combine the flux power spectrum taken along each axis. We have two scripts that will complete this so that we have one flux power spectrum to visualize.

Given some differential step in logarithmic $k$-space `dlogk`, we can take the largest Hubble flow along all axes in order to drive to the smallest $k$ value. We start with the smallest $k$ value and create a $k$-mode array that takes logarithmic steps like `dlogk`. We then loop over every ``k_x``, ``k_y``, and ``k_z`` value to find where they land on this new $k$-mode array, and use it to add the flux power spectrum from some axis, while normalizing by the number of number of $k$ values that land there. This can be done with the following call of the appropriate Python script

```bash
$ python3 group_powspec_dlogk.py $FPSOPTDEPTHBIN $DLOGK -v
```

where `$FPSOPTDEPTHBIN` is the output file from `powspec_skewer_quantiles.py`, `$DLOGK` is the differential step in log $k$-space, and the `-v` is a verbosity flag. The root group to the `$FPSOPTDEPTHBIN` file will gain a `dlogk` float and a `k_edges_dlogk` array that will hold the edges of the new $k$-mode array. Each ``FluxPowerSpectrum_quantile`` group also gains a `FPS_dlogk` dataset.


On the other hand, we can just group flux power spectra by the unique $k$-mode values in ``k_x``, ``k_y``, and ``k_z``. This is done with the appropriate Python script

```bash
$ python3 group_powspec_uniquek.py $FPSOPTDEPTHBIN -v
```

where the arguments are the same as before. The root group to the `$FPSOPTDEPTHBIN` file will gain a `k_uniq` array that will hold the unique $k$-mode values, while each ``FluxPowerSpectrum_quantile`` group gains a `FPS_uniq` dataset.



## Visualizing Quantiles

Now that we have one singular flux power spectra for some quantile that has been reduced from all three axes, we can plot them! We can plot all flux power spectra from each quantile where the line is colored by the mean effective optical depth. We call the plotting script as either

```bash
$ python3 plot_powspec_skewer_quantiles.py $FPSOPTDEPTHBIN -u -v -a
```

or 

```bash
$ python3 plot_powspec_skewer_quantiles.py $FPSOPTDEPTHBIN -g -v -a
```

where `$FPSOPTDEPTHBIN` is the output file from `powspec_skewer_quantiles.py`, `-v` is a verbosity flag, and `-a` tells the script to save plots of individual quantiles. Using either the `-u` flag or the `-d` flag tells the script to either use the flux power spectra grouped by unique $k$-mode values or by differential step in log $k$ space. These _could_ be made into seperate scripts, but ion wanna do that.


## Testing Against Cholla Analysis Files

We can test whether the transmitted flux power spectrum methods are consistent with On-The-Fly flux power spectrum by setting very large extremes in the optical depth ranges (say `1e-10` and `1e10`) with only one quantile (that is `NQUANTILE=1`). While the dlogk used are similar, we can test by plotting the relative difference from our flux power spectrum with the Cholla On-The-Fly results

```bash
$ python3 plot_powspec_diff.py $FPSOPTDEPTHBIN $CHOLLAOTFANALYSIS -v
```

where `$FPSOPTDEPTHBIN` is the output file from `powspec_skewer_quantiles.py`, `$CHOLLAOTFANALYSIS` is the analysis file from the On-The-Fly calculation, and `-v` is a verbosity flag. The output is a plot showing the relative difference. The `-l` flag will plot the relative difference in log-space.

## TODO

The quantile method is an effective way to get a good first look at the distribution of the transmitted flux power spectrum using skewers of different effective optical depth, but makes it difficult to test with different expansion histories because the effective optical depths at boundaries aren't set. My next immediate steps will be to make a script that will do the grouping assuming one quantile and just using one effective optical depth range. These would be appended on top of each other to have many ranges.

Test these scripts with a non-cube cosmological simulation.

In the future, I would like to somehow be able to bin by number of quantiles for multiple skewer files. I would also like to change how I show more than one skewer file on `plot_skewerdir_efftau.py`











