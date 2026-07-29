import argparse
from pathlib import Path
# from mpi4py import MPI

import numpy as np
import h5py


import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("dstyle")
_ = plt.figure()


###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the number of nodes
        and the parameter text file. Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Compute and append optical depth")

    parser.add_argument("datadirname", help='Cholla data output directory name', type=str)

    parser.add_argument('-r', '--restart', help='Reset progress bool array', 
                        action='store_true')

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    parser.add_argument('-x', '--xskewers', help='Skewers along x direction? (default: False)', 
                        action='store_true')

    parser.add_argument('-y', '--yskewers', help='Skewers along y direction? (default: False)', 
                        action='store_true')

    parser.add_argument('-z', '--zskewers', help='Skewers along z direction? (default: False)', 
                        action='store_true')

    return parser

###
# Create all data structures to fully explain optical depth calculation
# These data structures are pretty thorough, and not every line is readily needed
# but I prioritize readability over less lines of code
###


###
# Data structures related to hydro cholla data files
###

# Box-specific information that interacts with native hydro cholla data files

class ChollaBoxHead:
    '''
    Cholla Box Head object
        Holds information regarding the location of the data

        Initialized with:
        - nBox (int): number of the box within snapshot
        - cell_offset_x (int): number of cells offset in x dimension 
        - cell_offset_y (int): number of cells offset in y dimension
        - cell_offset_z (int): number of cells offset in z dimension
        - local_dimx (int): number of cells in the box subvolume in x dimension
        - local_dimy (int): number of cells in the box subvolume in y dimension
        - local_dimz (int): number of cells in the box subvolume in z dimension
    '''

    def __init__(self, nBox, cell_offset_x, cell_offset_y, cell_offset_z, local_dimx, local_dimy, local_dimz):
        self.nBox = nBox
        self.offset = (cell_offset_x, cell_offset_y, cell_offset_z)
        self.local_dims = (local_dimx, local_dimy, local_dimz)

    def set_coords(self, xmin, ymin, zmin, dx, dy, dz):
        '''
        Set the domain coordinates based on some cell size and min
        
        Args:
            xmin (float): minimum x coordinate
            ymin (float): minimum y coordinate
            zmin (float): minimum z coordinate
            dx (float): cell size in x dimension
            dy (float): cell size in y dimension
            dz (float): cell size in z dimension
        Returns:
            ...
        '''

        # set the local min/maximum x coordinate of the box
        self.local_xmin = xmin + (self.offset[0] * dx)
        self.local_xmax = xmin + ((self.offset[0] + self.local_dims[0]) * dx)

        self.local_ymin = ymin + (self.offset[1] * dy)
        self.local_ymax = ymin + ((self.offset[1] + self.local_dims[1]) * dy)

        self.local_zmin = zmin + (self.offset[2] * dz)
        self.local_zmax = zmin + ((self.offset[2] + self.local_dims[2]) * dz)


def greatest_prime_factor(num):
    '''
    Find the greatest prime factor of a number
    
    Args:
        num (int): number to factorize
    Returns:
        prime_factor (int): largest prime factor
    '''

    assert type(num) == int

    if ((num == 1) or (num == 2)):
        return num

    prime_factor = 2

    while (True):

        # keep dividing while evenly divisible
        while ((num % prime_factor) == 0):
            num /= prime_factor

        # cannot divide anymore
        if (num == 1):
            break

        # iterate to the next prime factor
        prime_factor += 1

    return prime_factor



class ChollaGrid:
    '''
    Cholla Grid object
        Holds global information about the domain and bounds of a simulation
            run. To complete domain decomposition, also need the number of
            processes simulation was ran with

        Initialized with:
        - nprocs (int): number of MPI processes
        - nx (int): number of cells in x dimension
        - ny (int): number of cells in y dimension
        - nz (int): number of cells in z dimension
        - xmin (float): minimum x coordinate
        - ymin (float): minimum y coordinate
        - zmin (float): minimum z coordinate
        - xmax (float): maximum x coordinate
        - ymax (float): maximum y coordinate
        - zmax (float): maximum z coordinate
    
    '''

    def __init__(self, nprocs, nx, ny, nz, xmin, ymin, zmin, xmax, ymax, zmax):

        if nprocs > 1:
            # odd number of processors not supported
            assert not nprocs % 2

        self.nprocs = nprocs

        # set number of cells globally
        self.nx_global, self.ny_global, self.nz_global = nx, ny, nz

        # set min, length, and size of cell in each dimension
        self.xmin = xmin
        self.Lx = xmax - self.xmin
        self.dx = self.Lx / self.nx_global

        self.ymin = ymin
        self.Ly = ymax - self.ymin
        self.dy = self.Ly / self.ny_global

        self.zmin = zmin
        self.Lz = zmax - self.zmin
        self.dz = self.Lz / self.nz_global

        # perform domain decomposition
        self.domain_decomp()


    def tile_decomposition(self):
        '''
        Tile the MPI processes in a block arrangement. Set the number
            of processes in each dimension as attributes. Assume 3D

        Args:
            ...
        Returns:
            ...
        '''
        np_x, np_y, np_z = 1, 1, 1
        nproc_tmp = int(self.nprocs)

        # get greatest prime factor of number of MPI processes
        n_gpf = greatest_prime_factor(nproc_tmp)

        index = 0
        while (nproc_tmp > 1):
            n_gpf = greatest_prime_factor(nproc_tmp)
            nproc_tmp = nproc_tmp // n_gpf

            if ((index % 3) == 0):
                np_x = int(np_x * n_gpf)
            elif ((index % 3) == 1):
                np_y = int(np_y * n_gpf)
            else:
                np_z = int(np_z * n_gpf)

            index += 1

        # ensure nx > ny > nz order
        if (np_z > np_y):
            temp = np_y
            np_y = np_z
            np_z = temp
        if (np_y > np_x):
            temp = np_x
            np_x = np_y
            np_y = temp
        if (np_z > np_y):
            temp = np_y
            np_y = np_z
            np_z = temp

        self.nproc_x = np_x
        self.nproc_y = np_y
        self.nproc_z = np_z

        return

    def domain_decomp(self):
        '''
        Tile the processes, set local subdomain sizes, and set starting 
            coords as attributes
    
        WARNING: assumes even split of global bounds
        
        Args:
            ...
        Returns:
            ...
        '''

        self.tile_decomposition()

        # set local x
        n = self.nx_global % self.nproc_x
        if (not n):
            self.nx_local = self.nx_global // self.nproc_x

        # set local y
        n = self.ny_global % self.nproc_y
        if (not n):
            self.ny_local = self.ny_global // self.nproc_y

        # set local z
        n = self.nz_global % self.nproc_z
        if (not n):
            self.nz_local = self.nz_global // self.nproc_z

        # set indices for each process
        n = 0
        self.index_x = np.zeros(self.nprocs, dtype=int)
        self.index_y = np.zeros(self.nprocs, dtype=int)
        self.index_z = np.zeros(self.nprocs, dtype=int)

        for k in range(self.nproc_z):
            for j in range(self.nproc_y):
                for i in range(self.nproc_x):
                    self.index_x[n] = i
                    self.index_y[n] = j
                    self.index_z[n] = k
                    n += 1

        return

    def get_BoxHead(self, nBox):
        '''
        Return the ChollaBoxHead object of this process number

        Args:
            nBox (int): process ID to use
        Return:
            boxhead (ChollaBoxHead): BoxHead for this object
        '''

        # calculate number of cell offsets
        offset_x = self.index_x[nBox] * self.nx_local
        offset_y = self.index_y[nBox] * self.ny_local
        offset_z = self.index_z[nBox] * self.nz_local

        # create object
        boxhead = ChollaBoxHead(nBox, offset_x, offset_y, offset_z,
                                self.nx_local, self.ny_local,
                                self.nz_local)
        # assign coordinates
        boxhead.set_coords(self.xmin, self.ymin, self.zmin, self.dx,
                               self.dy, self.dz)

        return boxhead

    def get_boxnum_ijk(self, i, j, k):
        '''
        Return the box number that an ijk cell resides in

        Args:
            i (int): x-dimension of cell
            j (int): y-dimension of cell
            k (int): z-dimension of cell
        Returns:
            (int): number of Box
        '''

        x_decomp = i // self.nx_local
        y_decomp = j // self.ny_local
        z_decomp = k // self.nz_local

        y_offset = y_decomp * self.nproc_x
        z_offset = z_decomp * self.nproc_x * self.nproc_y

        return int(x_decomp + y_offset + z_offset)


    def get_BoxHead_ijk(self, i, j, k):
        '''
        Return the ChollaBoxHead object that cell with ijk coordinates

        Args:
            i (int): x-dimension of cell
            j (int): y-dimension of cell
            k (int): z-dimension of cell
        Return:
            boxhead (ChollaBoxHead): BoxHead for this object
        '''

        nBox = self.get_boxnum_ijk(i,j,k)
        boxhead = self.get_BoxHead(nBox)

        return boxhead



# Calculations + bookkeeping related to cosmology and snapshot
class ChollaCosmologyLmtHead:
    '''
    Cholla Cosmology Head
        Serves as a header object that holds information that helps define a
            specific cosmology. Is limiting in the sense that it only holds
            OmegaM and OmegaL information
        
        Initialized with:
        - OmegaM (float): present-day energy density parameter for matter
        - OmegaL (float): present-day energy density parameter for dark energy
        - H0 (float): present-day Hubble parameter in units of [km / s / Mpc]
    '''

    def __init__(self, OmegaM, OmegaL, H0):

        # start with constants !
        self.Msun_cgs = 1.98847e33 # Solar Mass in grams
        self.kpc_cgs = 3.0857e21 # kiloparsecs in centimeters
        self.Mpc_cgs = self.kpc_cgs * 1.e3 # Megaparsecs in centimeters
        self.km_cgs = 1.e5 # kilometers in centimeters
        self.kyr_cgs = 3.15569e10 # kilo-years in seconds
        self.Myr_cgs = self.kyr_cgs * 1.e3 # mega-years in seconds
        self.Gyr_cgs = self.Myr_cgs * 1.e3 # giga-years in seconds

        self.G_cgs = 6.67259e-8 # gravitational constant in cgs [cm3 g-1 s-2]
        self.G_cosmo = self.G_cgs / self.km_cgs / self.km_cgs / self.kpc_cgs * self.Msun_cgs # gravitational constant in cosmological units [kpc (km2 s-2) Msun-1]
        self.kpc3_cgs = self.kpc_cgs * self.kpc_cgs * self.kpc_cgs
        self.Mpc3_cgs = self.Mpc_cgs * self.Mpc_cgs * self.Mpc_cgs

        # present-day energy density for matter, radiation, curvature, and Dark Energy
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL

        # present-day hubble parameter
        self.H0 = H0 # in [km s-1 Mpc-1]
        self.H0_cgs = self.H0 * self.km_cgs / self.Mpc_cgs # in cgs [s-1]
        self.H0_cosmo = self.H0 / 1.e3 # in cosmological units [km s-1 kpc-1]

        # dimensionless hubble parameter
        self.h_cosmo = self.H0 / 100.



###
# Data structures relating how OTF skewers are indexed (with some global id --
# the index of a skewer along some axis) and how the data is saved in native
# hydro cholla data files. With the parallelization of cholla, it's easy to
# think of a global skewer lying on a "face" that some process has access
# to and having a "local" id within this face
###

class ChollaSkewerLocalFaceHead:
    '''
    Cholla Skewer Local Face Head

    Holds information regarding a local skewer within face

        Initialized with:
        - localface_id (int): id of the local skewer on face
        - localface_joffset (int): offset along j-axis
        - localface_koffset (int): offset along k-axis
    '''
    def __init__(self, localface_id, localface_joffset, localface_koffset):
        self.localface_id = localface_id
        self.localface_joffset = localface_joffset
        self.localface_koffset = localface_koffset


class ChollaSkewerFaceHead:
    '''
    Cholla Skewer Face Head

    Holds information regarding a local skewer face

        Initialized with:
        - face_id (int): id of the face
        - face_joffset (int): offset along j-axis
        - face_koffset (int): offset along k-axis
    '''
    def __init__(self, face_id, face_joffset, face_koffset):
        self.face_id = face_id
        self.face_joffset = face_joffset
        self.face_koffset = face_koffset



class ChollaSkewerGlobalHead:
    '''
    Cholla Skewer Global Head

    Holds information regarding a global skewer

        Initialized with:
        - global_id (int): id of the global skewer
        - chFaceHead (ChollaSkewerFaceHead): ChollaSkewerFaceHead object,
            holds info on face within grid
        - chFaceLocalHead (ChollaSkewerLocalFaceHead): ChollaSkewerLocalFaceHead
            object, holds info on skewer within face
        - n_los (int): number of cells along line-of-sight
        - nlos_proc (int): number of processes along line-of-sight
    '''
    def __init__(self, global_id, ChollaSkewerFaceHead, ChollaSkewerLocalFaceHead, n_los, nlos_proc):
        self.global_id = global_id
        self.skewFaceHead = ChollaSkewerFaceHead
        self.skewLocalFaceHead = ChollaSkewerLocalFaceHead
        self.n_los = int(n_los)
        self.nlos_proc = int(nlos_proc)

    def get_globalj(self):
        '''
        Grab global j offset

        Args:
            ...
        Returns:
            (int): global j offset
        '''

        return int(self.skewFaceHead.face_joffset + self.skewLocalFaceHead.localface_joffset)

    def get_globalk(self):
        '''
        Grab global k offset

        Args:
            ...
        Returns:
            (int): global k offset
        '''

        return int(self.skewFaceHead.face_koffset + self.skewLocalFaceHead.localface_koffset)


###
# Data structures implementing a ChollaSkewerGlobalHead given some information
# related to the global grid of the simulation and how tight skewers are saved
# in On-The-Fly Skewers. Convention of xyz comes and memory management comes
# from Cholla's source code
###

class ChollaSkewerAnalysisHead:
    '''
    Cholla Skewer Analysis Head

    Holds information regarding a skewer analysis

        Initialized with:
        - nlos_global (int): number of line-of-sight global cells
        - nj_global (int): number of global cells along j-dimension
        - nk_global (int): number of global cells along k-dimension
        - nlos_proc (int): number of processes along line-of-sight
        - nj_proc (int): number of processes along j-dimension
        - nk_proc (int): number of processes along k-dimension
    '''
    def __init__(self, nlos_global, nj_global, nk_global, nlos_proc, nj_proc, nk_proc):
        self.nlos_global = nlos_global
        self.nlos_proc = nlos_proc
        self.nj_global = nj_global
        self.nk_global = nk_global

        self.ni_local = int(nlos_global / nlos_proc) # number of cells in process along los
        self.nj_local = int(nj_global / nj_proc) # number of cells in process along j-dimension
        self.nk_local = int(nk_global / nk_proc) # ^ along k-dimension

        self.nFaces_j, self.nFaces_k = int(nj_proc), int(nk_proc) # call number of processes a "face"
        self.nFaces_tot = int(self.nFaces_j * self.nFaces_k)

        self.nSkewersLocal = int(self.nj_local * self.nk_local)
        self.nSkewersTotal = self.nSkewersLocal * self.nFaces_tot

        assert self.nSkewersTotal == self.nj_global * self.nk_global


    def get_facehead_from_faceid(self, face_id):
        '''
        Return the ChollaSkewerFaceHead object corresponding to its face id
            Faces are tiled first along k-axis, then j-axis

        Args:
            face_id (int): id of the face
        Return:
            skewfacehead (ChollaSkewerFaceHead): FaceHead for this global skewer
        '''

        assert (0 <= face_id) and (face_id < self.nFaces_tot)

        face_joffset = int( (face_id % self.nFaces_j) * self.nj_local)
        face_koffset = int( (face_id // self.nFaces_j) * self.nk_local)

        skewfacehead = ChollaSkewerFaceHead(face_id, face_joffset, face_koffset)

        return skewfacehead

    def get_facehead_from_globalid(self, global_id):
        '''
        Return the ChollaSkewerFaceHead object corresponding to skewer global id
            Faces are tiled first along k-axis, then j-axis

        Args:
            global_id (int): id of the global skewer
        Return:
            skewfacehead (ChollaSkewerFaceHead): FaceHead for this global skewer
        '''

        assert (0 < global_id) and (global_id < self.nSkewersTotal)

        face_id = int(global_id // self.nSkewersLocal)
        face_joffset = int( (face_id % self.nFaces_j) * self.nj_local)
        face_koffset = int( (face_id // self.nFaces_j) * self.nk_local)

        skewfacehead = ChollaSkewerFaceHead(face_id, face_joffset, face_koffset)

        return skewfacehead

    def get_localfacehead_from_localid(self, local_id):
        '''
        Return the ChollaSkewerLocalFaceHead object corresponding to skewer 
            local id
            Local skewers are tiled first along k-axis, then j-axis

        Args:
            local_id (int): id of the local skewer
        Return:
            skewlocalhead (ChollaSkewerLocalFaceHead): LocalHead for this skewer
        '''

        assert (0 <= local_id) and (local_id < self.nSkewersLocal)

        local_joffset = int( (local_id % self.nSkewerslocal_j) )
        local_koffset = int( (local_id // self.nSkewerslocal_j) )

        skewlocalhead = ChollaSkewerLocalFaceHead(local_id, local_joffset,
                                                  local_koffset)

        return skewlocalhead

    def get_localfacehead_from_globalid(self, global_id):
        '''
        Return the ChollaSkewerLocalFaceHead object corresponding to skewer 
            global id
            Local skewers are tiled first along k-axis, then j-axis

        Args:
            global_id (int): id of the global skewer
        Return:
            skewlocalhead (ChollaSkewerLocalFaceHead): LocalHead for this global skewer
        '''

        assert (0 < global_id) and (global_id < self.nSkewersTotal)

        local_id = int(global_id % self.nSkewersLocal)
        local_joffset = int( (local_id % self.nSkewerslocal_j) )
        local_koffset = int( (local_id // self.nSkewerslocal_j) )

        skewlocalhead = ChollaSkewerLocalFaceHead(local_id, local_joffset,
                                                  local_koffset)

        return skewlocalhead

    def get_globalhead(self, global_id):
        '''
        Return the ChollaSkewerGlobalHead object corresponding to skewer global id

        Args:
            global_id (int): id of the global skewer
        Return:
            skewglobalhead (ChollaSkewerGlobalHead): GlobalHead for this global 
                skewer id
        '''

        assert (0 < global_id) and (global_id < self.nSkewersTotal)

        facehead = self.get_facehead_from_globalid(global_id)
        localhead = self.get_localfacehead_from_globalid(global_id)

        skewglobalhead = ChollaSkewerGlobalHead(global_id, facehead, localhead,
                                                self.nlos_global, self.nlos_proc)

        return skewglobalhead



class ChollaSnapCosmologyHead:
    '''
    Cholla Snapshot Cosmology header object
        Serves as a header holding information that combines a ChollaCosmologyHead
            with a specific scale factor with the snapshot header object.
        
        Initialized with:
            scale_factor (float): scale factor
            cosmoHead (ChollaCosmologyHead): provides helpful information of cosmology & units

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, scale_factor, cosmoHead):
        self.a = scale_factor
        self.cosmoHead = cosmoHead

        # calculate & attach current Hubble rate in [km s-1 Mpc-1] and [s-1]
        self.Hubble_cosmo = self.Hubble()
        self.Hubble_cgs = self.Hubble_cosmo * self.cosmoHead.km_cgs / self.cosmoHead.Mpc_cgs # in cgs [s-1]


    def Hubble(self):
        '''
        Return the current Hubble parameter

        Args:
            ...
        Returns:
            H (float): Hubble parameter (km/s/Mpc)
        '''
        w0, wa = -1. , 0.
        OmegaR, OmegaK = 0., 0.

        a2 = self.a * self.a
        a3 = a2 * self.a
        a4 = a3 * self.a
        DE_factor = (self.a)**(-3. * (1. + w0 + wa))
        DE_factor *= np.exp(-3. * wa * (1. - self.a))

        H0_factor = (OmegaR / a4) + (self.cosmoHead.OmegaM / a3)
        H0_factor += (OmegaK / a2) + (self.cosmoHead.OmegaL * DE_factor)

        return self.cosmoHead.H0 * np.sqrt(H0_factor)



def main():
    '''
    Append the array of optical depth of mean flux for a skewer file
    '''


    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    # rank_idstr = f"Rank {rank:.0f}"

    # if rank == 0:
    #     # Create parser 
    #     parser = create_parser()
    #     # Save args
    #     args = parser.parse_args()
    #     if args.verbose:
    #         print(f"--- We are looking at data file : {args.datadirname} ---")
    #         print(f"--- {rank_idstr} : Args parsed and created ! ---")
    # else:
    #     args = None


    # # Give args to all ranks
    # args = comm.bcast(args, root=0)

    # if args.verbose and rank == 0:
    #     print(f"{rank_idstr} : Using {size:.0f} processes !")
    #     print(f"--- {rank_idstr} : Args have been broadcasted! ---")

    parser = create_parser()
    #     # Save args
    args = parser.parse_args()
    assert args.xskewers or args.yskewers or args.zskewers

    # I get overwhelmed and can only do one at a time, pls be kind
    assert (args.xskewers + args.yskewers + args.zskewers) == 1


    # define where file name will be placed
    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        outdir_dirPath = Path.cwd()

    if args.verbose:
        if args.outdir:
            print(f"--- We are saving the plot at directory : {outdir_dirPath} ---")
        else:
            print(f"--- No output directory detailed, so it will be placed as : {outdir_dirPath} ---")

    precision = np.float64


    # Convert argument input to Path() & get its absolute path
    data_dirPath = Path(args.datadirname).resolve()
    data_nOutput = int(data_dirPath.name)
    data_initinfo_fPath = data_dirPath / Path(f"{data_nOutput}.h5.0")
    assert data_initinfo_fPath.is_file()


    # grab information to create Cholla Grid information - number of processes and bounds
    nprocsarr_key = 'nprocs'
    boundsarr_key = 'bounds'
    domainarr_key = 'domain'
    dimsarr_key = 'dims'

    # with h5py.File(data_initinfo_fPath, 'r', driver='mpio', comm=comm) as fObj:
    with h5py.File(data_initinfo_fPath, 'r') as fObj:
        nprocs_arr = fObj.attrs.get(nprocsarr_key)
        bounds = fObj.attrs.get(boundsarr_key)
        domain = fObj.attrs.get(domainarr_key)
        ni_arr = fObj.attrs.get(dimsarr_key)
        redshift = fObj.attrs['Current_z'].item()
        OmegaM = fObj.attrs.get('Omega_M').item()
        OmegaL = fObj.attrs.get('Omega_L').item()
        H0 = fObj.attrs.get('H0').item()

        # grab length of box in units of [kpc]
        Lbox_arr = np.array(fObj.attrs['domain'])

        Lx, Ly, Lz = Lbox_arr
        nx, ny, nz = ni_arr
        dx, dy, dz = (Lx / nx).item(), (Ly / ny).item(), (Lz / nz).item()

    nprocs_tot = np.prod(nprocs_arr)

    if args.yskewers:
        skew_axis = 1
    elif args.zskewers:
        skew_axis = 2
    else:
        skew_axis = 0

    nlos_proc = nprocs_arr[skew_axis]
    nlos = ni_arr[skew_axis]
    if (skew_axis == 1):
        nj = ni_arr[0]
        nk = ni_arr[2]
        nj_proc = nprocs_arr[0]
        nk_proc = nprocs_arr[2]
    elif (skew_axis == 2):
        nj = ni_arr[0]
        nk = ni_arr[1]
        nj_proc = nprocs_arr[0]
        nk_proc = nprocs_arr[1]
    else:
        nj = ni_arr[1]
        nk = ni_arr[2]
        nj_proc = nprocs_arr[1]
        nk_proc = nprocs_arr[2]


    assert not (nlos % nlos_proc)
    assert not (nj % nj_proc)
    assert not (nk % nk_proc)

    ni_perproc = nlos / nlos_proc
    nj_perproc = nj / nj_proc
    nk_perproc = nk / nk_proc

    Li = Lbox_arr[0]
    Lj = Lbox_arr[1]
    Lk = Lbox_arr[2]


    # create a Grid object
    chGrid = ChollaGrid(nprocs_tot, ni_arr[0], ni_arr[1], ni_arr[2],
                        bounds[0], bounds[1], bounds[2],
                        domain[0], domain[1], domain[2])

    seed = 1337

    SkewAnalysis = ChollaSkewerAnalysisHead(nlos,nj,nk,
                                            nlos_proc, nj_proc, nk_proc)

    # # focus on j,k face
    # jkface_ID_arr = np.arange(nj_proc * nk_proc)
    # jkface_IDs_rank = np.argwhere((jkface_ID_arr % size) == rank).flatten()

    # print(rank, jkface_IDs_rank)
    # for jkface_ID in jkface_IDs_rank:
    #     i,j,k = 0, nj_perproc * (jkface_ID // nj_proc), nk_perproc * (jkface_ID % nj_proc)

    #     for ni_proc_curr in range(ni_proc):
    #         i_curr = ni_proc_curr * ni_perproc
    #         boxHead = chGrid.get_BoxHead_ijk(i_curr,j,k)
    #         print(boxHead.offset, boxHead.local_dims)



    # create cosmology and snapshot header
    chCosmoHead = ChollaCosmologyLmtHead(OmegaM, OmegaL, H0)

    scale_factor = 1. / (1. + redshift)
    snapCosmoHead = ChollaSnapCosmologyHead(scale_factor, chCosmoHead)
    # print(rank)

if __name__=="__main__":
    main()

