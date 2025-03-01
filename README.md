# cholla_lya_scripts

We would like to study the Lyman-alphs Forest in cosmological Cholla asimulations by calculating the optical depth of Lyman-alpha along skewers.

However, our previous [study](https://github.com/astrodiegog/cholla_lya_scripts/tree/speedup-study) on speeding up the calculation (because it scales poorly with more cells), focused on changing the optical depth calculation itself - only including cells whose thermal velocity is near that of the cell. However, this led to different optical depth values for different windows to go around the thermal velocity, which in turn created changes in the transmitted flux power spectrum. Details on the analysis of this study can be found [here](https://cholla-cosmo.readthedocs.io/en/latest/study_gaussianspeedup.html#study-gauss-speed). 

This new repository aims to leverage parallelization to use many processors to attack the problem instead of changing the calculation itself. When doing the calculation, we serially loop over each skewer, and calculate the local optical depth along the entire line-of-sight, instead of only near the thermal velocity of a cell. While reproducing the flux power spectrum for the on-the-fly analysis, this is very computationally intensive. To get around this, we note that the calculation of the optical depth is independent of other skewers -- many skewer optical depth calculations can be done in parallel.

The Python package [mpi4py](https://mpi4py.readthedocs.io) provides Python bindings for the message passing interface (MPI) standard which is common in many parallelized codes, including Cholla itself. Instead of using Python, we could write this calculation in C itself, but one step at a time bro chilllllll. From serially looping over each skewer individually,

```python
for nSkewerID in range(nSkewers):
    ...
    tau_local[nSkewerID] = tau
```

we assign specific skewer IDs for each processor

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


The default optical depth script works as usual, where the local optical depth is already saved in this repo

```bash
$ python3 optdepth.py $SKEWERFILE -v
```

The `-v` flag tells the script to be verbose and pring helpful info, while `$SKEWERFILE` is the HDF5 skewer output file.

On the other hand, the new script runs with your favorite MPI standard

```bash
$ mpirun -np $NUMNODES python3 $SKEWERFILE -v
```

where the `-np` flag specifies the number of processors to use in the calculation, specified here with `$NUMNODES`.

The outcome of this study can be found [here](https://cholla-cosmo.readthedocs.io/en/latest/study_gaussianspeedup_mpi.html#study-gauss-speed-mpi).



