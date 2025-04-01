"""
This script updates the effective optical depth calculation from median optical
    depth to the optical depth from the mean flux across a skewer.

Usage:
    $ mpirun -np 8 python3 optdepth_updatetaueff.py 0_skewers.h5 -v
"""

import argparse
from pathlib import Path
from mpi4py import MPI


import numpy as np
import h5py
from scipy.special import erf


###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the skewer file name. 
        Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Update effective optical depth")

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument('-r', '--restart', help='Reset progress bool array', 
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser

###
# Create all data structures to fully explain optical depth calculation
###



# Skewer-specific information that interacts with skewers for a given skewer file
# ChollaOnTheFlySkewers_iHead   --> Holds skewer group
# ChollaOnTheFlySkewers_i       --> Creates ChollaOnTheFlySkewer object
# ChollaOnTheFlySkewers         --> Creates ChollaOnTheFlySkewers_i object


class ChollaOnTheFlySkewers_iHead:
    '''
    Cholla On The Fly Skewers_i Head

    Holds information regarding a specific skewer hdf5 group

        Initialized with:
        - n_i (int): length of the skewers
        - n_j (int): length of first dimension spanning cube
        - n_k (int): lenth of second dimension spanning cube
        - n_stride (int): stride cell number between skewers
        - skew_key (str): string to access skewer
    '''
    def __init__(self, n_i, n_j, n_k, n_stride, skew_key):
        self.n_i = n_i
        self.n_j = n_j
        self.n_k = n_k
        self.n_stride = n_stride
        self.skew_key = skew_key

        # number of skewers, assumes nstride is same along both j and k dims
        self.n_skews = int( (self.n_j * self.n_k) / (self.n_stride * self.n_stride) )


class ChollaOnTheFlySkewers_i:
    '''
    Cholla On The Fly Skewers
    
    Holds skewer specific information to an output with methods to 
            access data for that output

        Initialized with:
        - ChollaOTFSkewersiHead (ChollaOnTheFlySkewers_iHead): header
            information associated with skewer
        - fPath (PosixPath): file path to skewers output
        - comm (mpi4py.MPI.Comm): communication context 

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, ChollaOTFSkewersiHead, fPath, comm):
        self.OTFSkewersiHead = ChollaOTFSkewersiHead
        self.fPath = fPath.resolve() # convert to absolute path
        assert self.fPath.is_file() # make sure file exists
        self.comm = comm

        self.set_keys() # set possible skewer keys


    def set_keys(self):
        '''
        Check skewer group to set the available keys

        Args:
            ...
        Returns:
            ...
        '''

        with h5py.File(self.fPath, 'r') as fObj:
            self.allkeys = set(fObj[self.OTFSkewersiHead.skew_key].keys())

    def check_datakey(self, data_key):
        '''
        Check if a requested data key is valid to be accessed in skewers file

        Args:
            data_key (str): key string that will be used to access hdf5 dataset
        Return:
            (bool): whether data_key is a part of expected data keys
        '''

        return data_key in self.allkeys


class ChollaOnTheFlySkewers:
    '''
    Cholla On The Fly Skewers
    
    Holds on-the-fly skewers specific information to an output with methods to 
            create specific skewer objects

        Initialized with:
        - fPath (PosixPath): file path to skewers output    
        - comm (mpi4py.MPI.Comm): communication context    

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, fPath, comm):
        self.OTFSkewersfPath = fPath.resolve() # convert to absolute path
        assert self.OTFSkewersfPath.is_file() # make sure file exists

        self.comm = comm

        self.xskew_str = "skewers_x"
        self.yskew_str = "skewers_y"
        self.zskew_str = "skewers_z"

        # set grid information (ncells, dist between cells, nstride)
        self.set_gridinfo()
        dx_Mpc = self.dx / 1.e3 # [Mpc]
        dy_Mpc = self.dy / 1.e3
        dz_Mpc = self.dz / 1.e3

        # set cosmology params
        self.set_cosmoinfo()

        # grab current hubble param & info needed to calculate hubble flow
        H = self.get_currH()  # [km s-1 Mpc-1]
        cosmoh = self.H0 / 100.

        # calculate proper distance along each direction
        dxproper = dx_Mpc * self.current_a / cosmoh # [h-1 Mpc]
        dyproper = dy_Mpc * self.current_a / cosmoh
        dzproper = dz_Mpc * self.current_a / cosmoh

        # calculate Hubble flow through a cell along each axis
        self.dvHubble_x = H * dxproper # [km s-1]
        self.dvHubble_y = H * dyproper
        self.dvHubble_z = H * dzproper

    def set_gridinfo(self, datalength_str='density'):
        '''
        Set grid information by looking at attribute of file object and shape of 
            data sets
        
        Args:
            datalength_str (str): (optional) key to dataset used to find the
                number of skewers and cells along an axis
        Returns:
            ...
        '''
        with h5py.File(self.OTFSkewersfPath, 'r', driver='mpio', comm=self.comm) as fObj:
            # grab length of box in units of [kpc]
            Lx, Ly, Lz = np.array(fObj.attrs['Lbox'])

            # set number of skewers and stride number along each direction 
            nskewersx, self.nx = fObj[self.xskew_str][datalength_str].shape
            nskewersy, self.ny = fObj[self.yskew_str][datalength_str].shape
            nskewersz, self.nz = fObj[self.zskew_str][datalength_str].shape

        # we know nskewers_i = (nj * nk) / (nstride_i * nstride_i)
        # so nstride_i = sqrt( (nj * nk) / (nskewers_i) )
        self.nstride_x = int(np.sqrt( (self.ny * self.nz)/(nskewersx) ))
        self.nstride_y = int(np.sqrt( (self.nz * self.nx)/(nskewersy) ))
        self.nstride_z = int(np.sqrt( (self.nx * self.ny)/(nskewersz) ))

        # save cell distance in each direction to later calculate hubble flow
        self.dx = Lx / self.nx
        self.dy = Ly / self.ny
        self.dz = Lz / self.nz

        return


    def set_cosmoinfo(self):
        '''
        Set cosmological attributes for this object

        Args:
            comm (mpi4py.MPI.Comm): communication context
        Returns:
            ...
        '''

        with h5py.File(self.OTFSkewersfPath, 'r', driver='mpio', comm=self.comm) as fObj:
            self.Omega_R = fObj.attrs['Omega_R'].item()
            self.Omega_M = fObj.attrs['Omega_M'].item()
            self.Omega_L = fObj.attrs['Omega_L'].item()
            self.Omega_K = fObj.attrs['Omega_K'].item()

            self.w0 = fObj.attrs['w0'].item()
            self.wa = fObj.attrs['wa'].item()

            self.H0 = fObj.attrs['H0'].item() # expected in km/s/Mpc
            self.current_a = fObj.attrs['current_a'].item()
            self.current_z = fObj.attrs['current_z'].item()
        
        return

    def get_currH(self):
        '''
        Return the Hubble parameter at the current scale factor

        Args:
            ...
        Returns:
            H (float): Hubble parameter (km/s/Mpc)
        '''

        a2 = self.current_a * self.current_a
        a3 = a2 * self.current_a
        a4 = a3 * self.current_a
        DE_factor = (self.current_a)**(-3. * (1. + self.w0 + self.wa))
        DE_factor *= np.exp(-3. * self.wa * (1. - self.current_a))

        H0_factor = (self.Omega_R / a4) + (self.Omega_M / a3)
        H0_factor += (self.Omega_K / a2) + (self.Omega_L * DE_factor)

        return self.H0 * np.sqrt(H0_factor)

    def get_skewersx_obj(self):
        '''
        Return ChollaOnTheFlySkewers_i object of the x-skewers

        Args:
            ...
        Return:
            OTFSkewerx (ChollaOnTheFlySkewers_i): skewer object
        '''

        OTFSkewersxHead = ChollaOnTheFlySkewers_iHead(self.nx, self.ny, self.nz,
                                                      self.nstride_x, self.xskew_str)

        OTFSkewerx = ChollaOnTheFlySkewers_i(OTFSkewersxHead, self.OTFSkewersfPath, self.comm)

        return OTFSkewerx

    def get_skewersy_obj(self):
        '''
        Return ChollaOnTheFlySkewers_i object of the y-skewers

        Args:
            ...
        Return:
            OTFSkewery (ChollaOnTheFlySkewers_i): skewer object
        '''

        OTFSkewersyHead = ChollaOnTheFlySkewers_iHead(self.ny, self.nx, self.nz,
                                                      self.nstride_y, self.yskew_str)

        OTFSkewery = ChollaOnTheFlySkewers_i(OTFSkewersyHead, self.OTFSkewersfPath, self.comm)

        return OTFSkewery

    def get_skewersz_obj(self):
        '''
        Return ChollaOnTheFlySkewers_i object of the z-skewers

        Args:
            ...
        Return:
            OTFSkewerz (ChollaOnTheFlySkewers_i): skewer object
        '''

        OTFSkewerszHead = ChollaOnTheFlySkewers_iHead(self.nz, self.nx, self.ny,
                                                      self.nstride_z, self.zskew_str)

        OTFSkewerz = ChollaOnTheFlySkewers_i(OTFSkewerszHead, self.OTFSkewersfPath, self.comm)

        return OTFSkewerz


#####
# Script specific functions
#####



def taucalc_update(OTFSkewers_i, comm, precision=np.float64, verbose=False):
    '''
    Calculate the effective optical depth for each skewer along an axis

    Args:
        OTFSkewers_i (ChollaOnTheFlySkewers_i): holds all skewer info along an axis
        comm (mpi4py.MPI.Comm): communication context
        precision (np type): (optional) numpy precision to use
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''

    skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key
    rank = comm.Get_rank()
    size = comm.Get_size()
    rank_idstr = f"Rank {rank:.0f}"

    with h5py.File(OTFSkewers_i.fPath, 'r+', driver='mpio', comm=comm) as fObj:
        if verbose:
            print(f"--- {rank_idstr} : Starting calculations at along ", OTFSkewers_i.OTFSkewersiHead.skew_key, "---")

        skewerID_arr = np.arange(OTFSkewers_i.OTFSkewersiHead.n_skews)
        skewerIDs_rank = np.argwhere((skewerID_arr % size) == rank).flatten()

        # loop over each skewer
        for nSkewerID in skewerIDs_rank:
            # grab skewer data & calculate effective optical depth
            taus = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get('taucalc_local')[nSkewerID, :]

            fluxes = np.exp(- taus)
            meanF = np.mean(fluxes)

            # update eff tau arr for this skewer
            fObj[skew_key]['taucalc_eff'][nSkewerID] = -1. * np.log(meanF)


    if verbose:
        print(f"--- {rank_idstr} : Effective optical depth calculation completed along ", OTFSkewers_i.OTFSkewersiHead.skew_key)

    return



def main():
    '''
    Update the array of optical depth of mean flux for a skewer file
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rank_idstr = f"Rank {rank:.0f}"

    if rank == 0:
        # Create parser 
        parser = create_parser()
        # Save args
        args = parser.parse_args()
        if args.verbose:
            print(f"--- {rank_idstr} : Args parsed and created ! ---")
    else:
        args = None


    # Give args to all ranks
    args = comm.bcast(args, root=0)

    if args.verbose and rank == 0:
        print(f"{rank_idstr} : Using {size:.0f} processes !")
        print(f"--- {rank_idstr} : Args have been broadcasted! ---")


    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()

    if args.verbose:
        print(f"--- {rank_idstr} : skewer file {skewer_fPath} is a real file ! ---")

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath, comm)
    
    if args.verbose:
        print(f"--- {rank_idstr} : Cosmo Head and OTF Skewer objects created ---")
        print(f"--- {rank_idstr} : Making sure effective and local optical depth is inside skewers ---")

    optdeptheff_key = 'taucalc_eff'
    optdepthloc_key = 'taucalc_local'

    OTFSkewers_x = OTFSkewers.get_skewersx_obj()
    assert OTFSkewers_x.check_datakey(optdeptheff_key)
    assert OTFSkewers_x.check_datakey(optdepthloc_key)

    OTFSkewers_y = OTFSkewers.get_skewersy_obj()
    assert OTFSkewers_y.check_datakey(optdeptheff_key)
    assert OTFSkewers_y.check_datakey(optdepthloc_key)

    OTFSkewers_z = OTFSkewers.get_skewersz_obj()
    assert OTFSkewers_z.check_datakey(optdeptheff_key)
    assert OTFSkewers_z.check_datakey(optdepthloc_key)

    if args.verbose:
        print(f"--- {rank_idstr} : OTFSkewers in each axis are created ---")

    t_start = MPI.Wtime()

    taucalc_update(OTFSkewers_x, comm, precision, args.verbose)
    if args.verbose:
        t_x = MPI.Wtime()
        print(f"--- {rank_idstr} : Took {t_x - t_start:.4e} secs to update tau along x ---")


    taucalc_update(OTFSkewers_y, comm, precision, args.verbose)
    if args.verbose:
        t_y = MPI.Wtime()
        print(f"--- {rank_idstr} : Took {t_y - t_x:.4e} secs to update tau along y ---")

    taucalc_update(OTFSkewers_z, comm, precision, args.verbose)
    if args.verbose:
        t_z = MPI.Wtime()
        print(f"--- {rank_idstr} : Took {t_z - t_y:.4e} secs to update tau along z ---")

    t_end = MPI.Wtime()


    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+', driver='mpio', comm=comm) as fObj:
        if args.verbose:
            print(f"--- {rank_idstr} : Took {t_end - t_start:.4e} secs for entire calculation ---")

        fObj[f'updatetime_{size:.0f}_nprocs'][rank] = t_end - t_start

        

if __name__=="__main__":
    main()

