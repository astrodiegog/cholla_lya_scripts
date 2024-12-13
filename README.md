# cholla_lya_scripts

Python scripts to study the Lyman-alpha Forest in cosmological Cholla simulations

We would like to study the optical depth and transmitted flux power spectrum in a cosmological Cholla simulation. Our end goal will be a flux power spectrum, which comes from two inputs:

1. On-The-Fly Skewer File
2. Differential log-space to group Fourier Transform bins

We first run `optdepth.py` to calculate the optical depth along a skewer, then `powspec.py` to calculate the transmitted flux power spectrum.

What is the expected contents in the On-The-Fly Skewer File? 3 datasets for each of `skewers_x`, `skewers_y`, and `skewers_z` groups:

1. ``HI_density`` - ionized Hydrogen in comoving density units of $h^2 \textrm{M}\_{\odot} \textrm{kpc}^3$ 
2. ``los_velocity`` - line-of-sight peculiar velocity along a skewer in units of $\textrm{km} \textrm{s}^{-1}$ 
3. ``temperature`` - temperature in units of $\textrm{K}$ 

Each dataset is expected to be in shape of the number of skewers and line-of-sight cells $(n\_{\textrm{skewers}}, n\_{\textrm{LOS}})$.

10 attributes are also expected:

1. ``Lbox`` - length of simulated box in units of $\textrm{kpc}$
2. ``Omega_R`` - Present-Day Radiation Energy Density
3. ``Omega_M`` - Present-Day Matter Energy Density
4. ``Omega_L`` - Present-Day Dark Energy Density
5. ``Omega_K`` - Present-Day Spatial Curvature Energy Density
6. ``w0`` and ``wa`` - parameters specifying time-evolving Dark Energy equation of state
7. ``current_a`` and ``current_z`` - scale factor and redshift at which skewer data is taken
8. ``H0`` - Present-Day Hubble parameter in units of $\textrm{kpc} / \textrm{s} / \textrm{Mpc}$

The two scripts have been written with the python package [argparse](https://docs.python.org/3/howto/argparse.html) which allows for a quick command line interface.

To calculate the optical depth, we expect one positional argument of the Cholla skewer output file name, taking two optional parameters whether to store the local optical depth along a skewer and a verbose flag.

To calculate the optical depth, we run

```
$ python3 optdepth.py $SKEWERFILE -v -l
```

where 

``$SKEWERFILE`` is the one positional argument - the skewer output file
``-v`` flags the script to be verbose throughout the calculation
``-l`` flags the script to save the local optical depth

To only save the median optical depth, and not the local optical depth, do not include the ``-l`` flag.


To calculate the transmitted flux power spectrum, we run

```
$ python powspec.py $SKEWERFILE $DLOGK -v -c
```

where 

``$SKEWERFILE`` is the first positional argument - the skewer output file
``$DLOGK`` is the second positional argument - the differential log-space bin size
``-v`` flags the script to be verbose throughout the calculation
``-c`` flags the script to save the averaged power spectrum along all three axes



