"""
This script boosts the number density for all values when calculating the local
    optical depth to match a target optical depth. This script is adapted from
    optdepth_mpi.py to utilize parallelization. This "enhanced" optical depth
    value is saved as "taucalc_local_match" with the scalar multiplied to the 
    number density saved as "taumatch_HIdensity_scalar"

Usage:
    $ mpirun -np 8 python3 optdepth_match_mpi.py 0_skewers.h5 -v
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
        description="Compute and append optical depth")

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument("tau_target", help='Optical depth to match to', type=float)

    parser.add_argument('-r', '--restart', help='Print info along the way',
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser

###
# Create all data structures to fully explain optical depth calculation
###

###
# Calculations+bookkeeping related to cosmology, snapshot, and how optical depth is calculated
###
# ChollaCosmologyHead           --> cosmology-specific info
# ChollaSnapCosmologyHead       --> combines ChollaSnap+ChollaCosmo
# ChollaCosmoCalculator         --> calculator for cosmology snapshot (unit conversions)
# ChollaHydroCalculator         --> cgs constants & doppler param method (indpt of scale factor) 
# ChollaSkewerCosmoCalculator   --> implements optical depth calculation along skewer length


class ChollaCosmologyHead:
    '''
    Cholla Cosmology Head
        Serves as a header object that holds information that helps define a
            specific cosmology
        
        Initialized with:
        - OmegaM (float): present-day energy density parameter for matter
        - OmegaR (float): present-day energy density parameter for radiation
        - OmegaK (float): present-day energy density parameter for spatial curvature
        - OmegaL (float): present-day energy density parameter for dark energy
        - w0 (float): constant term in dark energy equation of state
        - wa (float): linear term in dark energy equation of state
        - H0 (float): present-day Hubble parameter in units of [km / s / Mpc]
    '''

    def __init__(self, OmegaM, OmegaR, OmegaK, OmegaL, w0, wa, H0):

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
        self.OmegaR = OmegaR
        self.OmegaK = OmegaK
        self.OmegaL = OmegaL

        # Dark Energy equation of state like w(a) = w0 + wa(1-a)
        self.w0, self.wa = w0, wa

        # present-day hubble parameter
        self.H0 = H0 # in [km s-1 Mpc-1]
        self.H0_cgs = self.H0 * self.km_cgs / self.Mpc_cgs # in cgs [s-1]
        self.H0_cosmo = self.H0 / 1.e3 # in cosmological units [km s-1 kpc-1]

        # dimensionless hubble parameter
        self.h_cosmo = self.H0 / 100.

        # Hubble time (1/H0)
        self.t_H0_cgs = 1. / self.H0_cgs # in seconds
        self.t_H0_gyrs = self.t_H0_cgs / self.Gyr_cgs # in Gyrs
        self.t_H0_cosmo  = self.t_H0_cgs * self.km_cgs / self.kpc_cgs # in cosmological units [s kpc km-1]

        # critical density in units of [g cm-3]
        self.rho_crit0_cgs = 3. * self.H0_cgs * self.H0_cgs / (8. * np.pi * self.G_cgs)

        # critical density in units of [h2 Msun kpc-3]
        self.rho_crit0_cosmo = self.rho_crit0_cgs * (self.kpc3_cgs) / (self.Msun_cgs) / self.h_cosmo / self.h_cosmo


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

        a2 = self.a * self.a
        a3 = a2 * self.a
        a4 = a3 * self.a
        DE_factor = (self.a)**(-3. * (1. + self.cosmoHead.w0 + self.cosmoHead.wa))
        DE_factor *= np.exp(-3. * self.cosmoHead.wa * (1. - self.a))

        H0_factor = (self.cosmoHead.OmegaR / a4) + (self.cosmoHead.OmegaM / a3)
        H0_factor += (self.cosmoHead.OmegaK / a2) + (self.cosmoHead.OmegaL * DE_factor)

        return self.cosmoHead.H0 * np.sqrt(H0_factor)

    def dvHubble(self, dx_h):
        '''
        Return the Hubble flow through a cell

        Args:
            dx_h (float): comoving distance between cells (h-1 kpc)
        Returns:
            (float): Hubble flow over a cell (km/s)
        '''
        # convert [h-1 kpc] to [kpc]
        dx = dx_h / self.cosmoHead.h_cosmo

        dx_cgs = dx * self.cosmoHead.kpc_cgs # kpc * (#cm / kpc) = cm
        dx_Mpc = dx_cgs / self.cosmoHead.Mpc_cgs # cm / (#cm / Mpc) = Mpc

        # convert to physical length
        dx_Mpc_phys = dx_Mpc * self.a

        return self.Hubble() * dx_Mpc_phys


class ChollaCosmoCalculator:
    '''
    Cholla Cosmological Calculator object
        Serves as a calculator for a cosmology at a specific scale factor.
        
        Initialized with:
            snapCosmoHead (ChollaSnapCosmologyHead): provides current redshift
            dims (tuple): size of data sets to act on
            dtype (np type): (optional) numpy precision to initialize output arrays 

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, snapCosmoHead, dims, dtype=np.float32):
        self.snapCosmoHead = snapCosmoHead
        self.dims = dims
        self.dtype = dtype

    def create_arr(self):
        '''
        Create and return an empty array
        
        Args:
            ...
        Returns:
            (arr): array of initialized dimensions and datatype
        '''

        return np.zeros(self.dims, dtype=self.dtype)

    def physical_length(self, length_comov):
        '''
        Calculate the physical length from a comoving length

        Args:
            length_comov (float): comoving length
        Returns:
            arr (arr): array that will hold data
        '''
        assert np.array_equal(length_comov.shape, self.dims)

        # initialize array with dims shape
        arr = self.create_arr()

        arr[:] = length_comov * self.snapCosmoHead.a

        return arr

    def physical_density(self, density_comov):
        '''
        Calculate the physical density from a comoving density

        Args:
            density_comov (float): comoving density
        Returns:
            arr (arr): array that will hold data
        '''
        assert np.array_equal(density_comov.shape, self.dims)

        # initialize array with dims shape
        arr = self.create_arr()

        a3 = self.snapCosmoHead.a * self.snapCosmoHead.a * self.snapCosmoHead.a
        arr[:] = density_comov / a3

        return arr

    def density_cosmo2cgs(self, density_cosmo):
        '''
        Convert the density saved in cosmological units of [h2 Msun kpc-3]
            to cgs units of [g cm-3]. With the large orders of magnitude
            involved, this calculation is completed in log-space

        Args:
            density_cosmo (float): density in cosmological units
        Returns:
            arr (arr): array that will hold data
        '''
        assert np.array_equal(density_cosmo.shape, self.dims)

        # initialize array with dims shape
        arr = self.create_arr()

        # calculate h^2
        h_cosmo2 = self.snapCosmoHead.cosmoHead.h_cosmo * self.snapCosmoHead.cosmoHead.h_cosmo

        # take log of constants
        ln_hcosmo2 = np.log(h_cosmo2)
        ln_Msun = np.log(self.snapCosmoHead.cosmoHead.Msun_cgs)
        ln_kpc3 = np.log(self.snapCosmoHead.cosmoHead.kpc3_cgs)

        # take log of density
        ln_density_cosmo = np.log(density_cosmo)

        # convert values to cgs
        ln_density_cgs = ln_density_cosmo + ln_Msun + ln_hcosmo2 - ln_kpc3

        # take exp of log to get physical values
        arr[:] = np.exp(ln_density_cgs) # [g cm-3]

        return arr

    def velocity_cosmo2cgs(self, velocity_cosmo):
        '''
        Convert the velocity saved in cosmology units of [km s-1] to the cgs
            units of [cm s-1].

        Args:
            velocity_cosmo (float): velocity in cosmological units
        Returns:
            arr (arr): array that will hold data
        '''
        assert np.array_equal(velocity_cosmo.shape, self.dims)

        # initialize array with dims shape
        arr = self.create_arr()

        arr[:] = velocity_cosmo * self.snapCosmoHead.cosmoHead.km_cgs # [cm s-1]

        return arr



class ChollaHydroCalculator:
    '''
    Cholla Calculator object
        Serves as a calculator where the calculated values have some expected
            size and datatype (default is float). Assert that inputs are of same
            shape as dims that was used to initialize this calculator. To 
            complete some analysis, this ChollaCalculator will be the mediator 
            that will act on the primitive saved values. 

        Initialized with:
            dims (tuple): size of data sets to act on
            dtype (np type): (optional) numpy precision to initialize output arrays
    
    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, dims, dtype=np.float32):
        self.dims = dims
        self.dtype = dtype

        # cgs constant values
        # proton and electron masses
        self.mp = 1.672622e-24 # [g]
        self.me = 9.1093897e-28 # [g]
        # Boltzmann constant
        self.kB = 1.380658e-16 # [erg K-1] = [cm2 g s-2 K-1]
        # electron charge
        self.e = 4.8032068e-10 # [esu] = [cm3/2 g1/2 s-1]
        # speed of light
        self.c = 2.99792458e10 # [cm s-1]
        # Ly-alpha wavelength
        self.lambda_Lya = 1.21567e-5 # [cm]

    def create_arr(self):
        '''
        Create and return an empty array
        
        Args:
            ...
        Returns:
            (arr): array of initialized dimensions and datatype
        '''

        return np.zeros(self.dims, dtype=self.dtype)

    def Doppler_param_Hydrogen(self, temp):
        '''
        Calculate the Doppler broadening parameter for distribution of Hydrogen
            in units of [cm s-1]

        Args:
            temp (arr): temperature of Hydrogen distribution
        '''

        assert np.array_equal(temp.shape, self.dims)

        # initialize array with dims shape
        arr = self.create_arr()

        arr[:] = np.sqrt(2. * self.kB * temp / self.mp)

        return arr



class ChollaSkewerCosmoCalculator:
    '''
    Cholla Skewer Calculator object
        Serves as a specific implementaiton of a Cholla Cosmological Calculator
            for a skewer.

        Initialized with:
            scale_factor (float): scale factor
            cosmoHead (ChollaCosmologyHead): provides helpful information of cosmology & units
            n_los (int): number of cells along line-of-sight
            dx_h (float): comoving distance between cells (h-1 kpc)
            dtype (np type): (optional) numpy precision to initialize output arrays
        
        Objects including ghost cells are suffixed with _ghost

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, scale_factor, cosmoHead, n_los, dx_h, dtype=np.float32):
        self.n_los = n_los
        self.n_ghost = int(0.1 * n_los) # take 10% from bruno
        self.dx_h = dx_h
        self.a = scale_factor

        # number of line-of-sight cells including ghost cells
        self.n_los_ghost = self.n_los + 2 * self.n_ghost

        # create ChollaCosmoCalc object
        self.snapCosmoHead = ChollaSnapCosmologyHead(self.a, cosmoHead)
        calc_dims, calc_dims_ghost = (self.n_los,), (self.n_los_ghost,)
        self.snapCosmoCalc = ChollaCosmoCalculator(self.snapCosmoHead, calc_dims, dtype=dtype)
        self.snapCosmoCalc_ghost = ChollaCosmoCalculator(self.snapCosmoHead, calc_dims_ghost, dtype=dtype)

        # create HydroCalc objects
        self.hydroCalc = ChollaHydroCalculator(calc_dims, dtype=dtype)
        self.hydroCalc_ghost = ChollaHydroCalculator(calc_dims_ghost, dtype=dtype)

        # calculate Hubble flow through one cell
        dvHubble = self.snapCosmoHead.dvHubble(self.dx_h) # [km s-1]
        self.dvHubble_cgs = dvHubble * self.snapCosmoHead.cosmoHead.km_cgs # [cm s-1]

        # create Hubble flow arrays along left, right, and center of each cell
        # prepend and append ghost cells
        self.vHubbleL_ghost_cgs = np.arange(-self.n_ghost, self.n_ghost + self.n_los) * self.dvHubble_cgs
        self.vHubbleR_ghost_cgs = self.vHubbleL_ghost_cgs + self.dvHubble_cgs
        self.vHubbleC_ghost_cgs = self.vHubbleL_ghost_cgs + 0.5 * self.dvHubble_cgs

    def extend_ghostcells(self, arr):
        '''
        Extend an array with ghost cels, enforcing periodic boundary conditions
        
        Args:
            arr (arr): array to extend
        Returns:
            arr_ghost (arr): extended array
        '''

        # create array with correct size, inherit arr datatype
        arr_ghost = np.zeros(self.n_los_ghost, dtype=arr.dtype)

        # copy over the real data
        arr_ghost[self.n_ghost : self.n_ghost + self.n_los] = arr[:]

        # extend far edge of arr to near edge of ghost array
        arr_ghost[ : self.n_ghost] = arr[-self.n_ghost : ]

        # extend near edge of arr to far edge of ghost array
        arr_ghost[-self.n_ghost : ] = arr[ : self.n_ghost]

        return arr_ghost


    def optical_depth_Hydrogen(self, densityHI, velocity_pec, temp):
        '''
        Compute the optical depth for each cell along the line-of-sight

        Args:
            densityHI (arr): ionized Hydrogen comoving density [h2 Msun kpc-3]
            velocity_pec (arr): peculiar velocity [km s-1]
            temp (arr): temperature [K]
        Returns:
            tau (arr): optical depth for each cell
        '''
        assert densityHI.size == self.n_los
        assert velocity_pec.size == self.n_los
        assert temp.size == self.n_los

        # convert comoving density to physical density then to cgs
        densityHI_phys = self.snapCosmoCalc.physical_density(densityHI)
        densityHI_phys_cgs = self.snapCosmoCalc.density_cosmo2cgs(densityHI_phys) # [g cm-3]

        # calculate column number density & extend to ghost cells
        nHI_phys_cgs = densityHI_phys_cgs / self.hydroCalc.mp # [cm-3]
        nHI_phys_ghost_cgs = self.extend_ghostcells(nHI_phys_cgs)

        # convert peculiar velocity to cgs values & extend to ghost cells
        velocity_pec_cgs = self.snapCosmoCalc.velocity_cosmo2cgs(velocity_pec)
        velocity_pec_ghost_cgs = self.extend_ghostcells(velocity_pec_cgs)
        # convert peculiar to physical velocity by adding Hubble flow
        velocity_phys_ghost_cgs = velocity_pec_ghost_cgs + self.vHubbleC_ghost_cgs # [cm s-1]

        # calculate doppler broadening param & extend to ghost cells
        doppler_param_cgs = self.hydroCalc.Doppler_param_Hydrogen(temp) # [cm s-1]
        doppler_param_ghost_cgs = self.extend_ghostcells(doppler_param_cgs)

        # calculate Ly-alpha interaction cross section
        sigma_Lya = np.pi * self.hydroCalc.e * self.hydroCalc.e / self.hydroCalc.me # [cm3 g1 s-2 / g] = [cm3 s-2]
        sigma_Lya = sigma_Lya * self.hydroCalc.lambda_Lya / self.hydroCalc.c # [cm3 s-2 * cm / (cm s-1)] = [cm3 s-1]
        sigma_Lya = sigma_Lya / self.snapCosmoHead.Hubble_cgs # [cm3 s-1 / (s-1)] = [cm3]
        f_12 = 0.416 # oscillator strength
        sigma_Lya *= f_12

        # initialize optical depths
        tau_ghost = self.snapCosmoCalc_ghost.create_arr()

        for losid in range(self.n_los_ghost):
            vH_L, vH_R = self.vHubbleL_ghost_cgs[losid], self.vHubbleR_ghost_cgs[losid]
            # calculate line center shift in terms of broadening scale
            y_L = (vH_L - velocity_phys_ghost_cgs) / doppler_param_ghost_cgs
            y_R = (vH_R - velocity_phys_ghost_cgs) / doppler_param_ghost_cgs
            # [cm3 * # density] = [cm3 * cm-3] = []
            tau_ghost[losid] = sigma_Lya * np.sum(nHI_phys_ghost_cgs * (erf(y_R) - erf(y_L))) / 2.0

        # clip edges
        tau = tau_ghost[self.n_ghost : -self.n_ghost]

        return tau


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
        Check skewer group to set the available keys. Keys in `skewer_i` groups
            may lead to a 1D or 2D dataset

        Args:
            ...
        Returns:
            ...
        '''

        keys_1D, keys_2D = [], []
        with h5py.File(self.fPath, 'r') as fObj:
            self.allkeys = set(fObj[self.OTFSkewersiHead.skew_key].keys())
            for key in self.allkeys:
                if fObj[self.OTFSkewersiHead.skew_key].get(key).ndim == 1:
                    keys_1D.append(key)
                if fObj[self.OTFSkewersiHead.skew_key].get(key).ndim == 2:
                    keys_2D.append(key)

        self.keys_1D = set(keys_1D)
        self.keys_2D = set(keys_2D)

        return

    def check_datakey(self, data_key):
        '''
        Check if a requested data key is valid to be accessed in skewers file

        Args:
            data_key (str): key string that will be used to access hdf5 dataset
        Return:
            (bool): whether data_key is a part of expected data keys
        '''

        return data_key in self.allkeys

    def get_skeweralldata(self, key, dtype=np.float32):
        '''
        Return a specific dataset for all skewers.
            Use this method with caution, as the resulting array can be large

            For (2048)^3 + nstride=4 + float64, resulting array will be ~4 GBs

        Args:
            key (str): key to access data from hdf5 file
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): requested dataset
        '''

        assert self.check_datakey(key)

        if key in self.keys_1D:
            arr = np.zeros((self.OTFSkewersiHead.n_skews), dtype=dtype)
            with h5py.File(self.fPath, 'r') as fObj:
                arr[:] = fObj[self.OTFSkewersiHead.skew_key].get(key)[:]
        elif key in self.keys_2D:
            arr = np.zeros((self.OTFSkewersiHead.n_skews, self.OTFSkewersiHead.n_i), dtype=dtype)
            with h5py.File(self.fPath, 'r') as fObj:
                arr[:,:] = fObj[self.OTFSkewersiHead.skew_key].get(key)[:, :]

        return arr



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
        dx_h_Mpc = self.dx_h_kpc / 1.e3 # [h-1 Mpc]
        dy_h_Mpc = self.dy_h_kpc / 1.e3
        dz_h_Mpc = self.dz_h_kpc / 1.e3

        # set cosmology params
        self.set_cosmoinfo()

        # grab current hubble param & info needed to calculate hubble flow
        H = self.get_currH()  # [km s-1 Mpc-1]
        cosmoh = self.H0 / 100.

        # calculate proper distance along each direction
        dxproper = dx_h_Mpc * self.current_a / cosmoh # [Mpc]
        dyproper = dy_h_Mpc * self.current_a / cosmoh
        dzproper = dz_h_Mpc * self.current_a / cosmoh

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
            # grab length of box in units of [h-1 kpc]
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
        self.dx_h_kpc = Lx / self.nx
        self.dy_h_kpc = Ly / self.ny
        self.dz_h_kpc = Lz / self.nz

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
            self.Omega_b = fObj.attrs['Omega_b'].item()

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


def init_taucalc(OTFSkewers, tau_target, nmax, comm, restart=False, verbose=False):
    '''
    Initialize the calculation of the effective optical depth matched. For each skewers_i axis
        group, create three things:
        1. (attribute) progress for optical depth
        2. (dataset) boolean mask whether optical depth has been calculated for 
            a specific skewer
        3. (dataset) optical depth of mean flux for a skewer
    
    Current implementation assumes the same nStride along each axis

    Args:
        OTFSkewers (ChollaOnTheFlySkewers): holds OTF skewers specific info
        tau_target (float): the optical depth seeking to match
        nmax (int): number of new density scalars to attempt
        comm (mpi4py.MPI.Comm): communication context
        restart (bool): (optional) whether to reset progress
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()
    rank_idstr = f"Rank {rank:.0f}"

    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+', driver='mpio', comm=comm) as fObj:

        if verbose:
            print(f'--- {rank_idstr} : \t...initializing optical depth calculations for file {OTFSkewers.OTFSkewersfPath} ---')

        OTF_attrs_keys = fObj.attrs.keys()

        density_scale_delta_max = 1.e8
        if 'density_scale_delta_max' not in OTF_attrs_keys:
            if verbose:
                print(f"--- {rank_idstr} : \t creating upper density scale delta ---")
            _ = fObj.attrs.create('density_scale_delta_max', density_scale_delta_max)
        elif restart:
            fObj.attrs['density_scale_delta_max'] = density_scale_delta_max
            if verbose:
                print(f"--- {rank_idstr} : \t restarting with upper density scale delta of {density_scale_delta_max:.4e} ---")
        else:
            if verbose:
                density_scale_delta_max = fObj.attrs['density_scale_delta_max'].item()
                print(f"--- {rank_idstr} : \t starting out with saved upper density scale delta of {density_scale_delta_max:.4e} ---")


        density_scale_delta_min = 1.e-8
        if 'density_scale_delta_min' not in OTF_attrs_keys:
            if verbose:
                print(f"--- {rank_idstr} : \t creating lower density scale delta ---")
            _ = fObj.attrs.create('density_scale_delta_min', density_scale_delta_min)
        elif restart:
            fObj.attrs['density_scale_delta_min'] = density_scale_delta_min
            if verbose:
                print(f"--- {rank_idstr} : \t restarting out with lower density scale delta of {density_scale_delta_min:.4e} ---")
        else:
            if verbose:
                density_scale_delta_min = fObj.attrs['density_scale_delta_min'].item()
                print(f"--- {rank_idstr} : \t starting out with saved lower density scale delta of {density_scale_delta_min:.4e} ---")


        if 'tau_target_match' not in OTF_attrs_keys:
            _ = fObj.attrs.create('tau_target_match', tau_target)
        elif not restart:
            assert fObj.attrs['tau_target_match'] == tau_target
        else:
            fObj.attrs['tau_target_match'] = tau_target

        OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                        OTFSkewers.get_skewersy_obj(),
                        OTFSkewers.get_skewersz_obj()]

        # add progress attribute, boolean mask for whether tau is calculated, and tau itself
        for i, OTFSkewers_i in enumerate(OTFSkewers_lst):

            if verbose:
                print(f"--- {rank_idstr} : \t\t...making sure optical depth already calculated along axis {i:.0f} ---")
            skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key
            OTFSkew_keys = fObj[skew_key].keys()

            # make sure optical depth has already been calculated
            assert 'taucalc_bool' in OTFSkew_keys
            assert np.sum(fObj[skew_key]['taucalc_bool']) == OTFSkewers_i.OTFSkewersiHead.n_skews

            if 'taucalc_eff_match' not in OTFSkew_keys:
                if verbose:
                    print(f"--- {rank_idstr} : \t\t...creating taucalc effective array along axis {i:.0f} ---")
                taucalc_eff = np.zeros(OTFSkewers_i.OTFSkewersiHead.n_skews, dtype=np.float64)
                fObj[skew_key].create_dataset('taucalc_eff_match', data=taucalc_eff)


            if 'taucalc_local_match' not in OTFSkew_keys:
                if verbose:
                    print(f"--- {rank_idstr} : \t\t...creating taucalc local array along axis {i:.0f} ---")
                taucalc_local = np.zeros((OTFSkewers_i.OTFSkewersiHead.n_skews, OTFSkewers_i.OTFSkewersiHead.n_i),
                                        dtype=np.float64)
                fObj[skew_key].create_dataset('taucalc_local_match', data=taucalc_local)



    # save snapshot specific information
    scale_factor = OTFSkewers.current_a
    redshift = OTFSkewers.current_z

    # save cosmology info
    Omega_K, Omega_L = OTFSkewers.Omega_K, OTFSkewers.Omega_L
    Omega_M, Omega_R = OTFSkewers.Omega_M, OTFSkewers.Omega_R
    Omega_b, H0 = OTFSkewers.Omega_b, OTFSkewers.H0
    w0, wa = OTFSkewers.w0, OTFSkewers.wa
    Lx_box = OTFSkewers.dx * OTFSkewers.nx
    Ly_box = OTFSkewers.dy * OTFSkewers.ny
    Lz_box = OTFSkewers.dz * OTFSkewers.nz
    Lbox = np.array([Lx_box, Ly_box, Lz_box])

    fName_split = OTFSkewers.OTFSkewersfPath.name.split('.')
    fName_match = fName_split[0] + "_taumatch." + fName_split[1]
    tauMatch_fPath = OTFSkewers.OTFSkewersfPath.parent / Path(fName_match)

    with h5py.File(tauMatch_fPath, 'w', driver='mpio', comm=comm) as fObj:
        if verbose:
            print(f'--- {rank_idstr} : \t...initializing optical depth calculations for file {tauMatch_fPath} ---')

        # start with cosmo info
        _ = fObj.attrs.create('Omega_R', Omega_R)
        _ = fObj.attrs.create('Omega_M', Omega_M)
        _ = fObj.attrs.create('Omega_L', Omega_L)
        _ = fObj.attrs.create('Omega_K', Omega_K)
        _ = fObj.attrs.create('Omega_b', Omega_b)
        _ = fObj.attrs.create('H0', H0)
        _ = fObj.attrs.create('w0', w0)
        _ = fObj.attrs.create('wa', wa)
        _ = fObj.attrs.create('current_a', scale_factor)
        _ = fObj.attrs.create('current_z', redshift)
        _ = fObj.attrs.create('tau_target', tau_target)
        _ = fObj.attrs.create('Lbox', Lbox)

        if f'calctime_{size:.0f}_nprocs_match' not in fObj.keys():
            calctime_arr = np.zeros(size, dtype=np.float64)
            fObj.create_dataset(f'calctime_{size:.0f}_nprocs_match', data=calctime_arr)

        if f'inittime_{size:.0f}_nprocs_match' not in fObj.keys():
            calctime_arr = np.zeros(size, dtype=np.float64)
            fObj.create_dataset(f'inittime_{size:.0f}_nprocs_match', data=calctime_arr)

        if f'taucalc_eff_history' not in fObj.keys():
            calctime_arr = np.zeros(nmax+1, dtype=np.float64)
            fObj.create_dataset(f'taucalc_eff_history', data=calctime_arr)

        if f'density_scale_history' not in fObj.keys():
            calctime_arr = np.zeros(nmax+1, dtype=np.float64)
            fObj.create_dataset(f'density_scale_history', data=calctime_arr)

        OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                        OTFSkewers.get_skewersy_obj(),
                        OTFSkewers.get_skewersz_obj()]

        # add progress attribute, boolean mask for whether tau is calculated, and tau itself
        for i, OTFSkewers_i in enumerate(OTFSkewers_lst):

            skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key
            skew_group = fObj.create_group(skew_key)
            OTFSkew_keys = fObj[skew_key].keys()

            if 'taucalc_eff' not in OTFSkew_keys:
                if verbose:
                    print(f"--- {rank_idstr} : \t\t...creating taucalc effective array along axis {i:.0f} ---")
                taucalc_eff = np.zeros(OTFSkewers_i.OTFSkewersiHead.n_skews, dtype=np.float64)
                fObj[skew_key].create_dataset('taucalc_eff', data=taucalc_eff)


            if 'taucalc_local' not in OTFSkew_keys:
                if verbose:
                    print(f"--- {rank_idstr} : \t\t...creating taucalc local array along axis {i:.0f} ---")
                taucalc_local = np.zeros((OTFSkewers_i.OTFSkewersiHead.n_skews, OTFSkewers_i.OTFSkewersiHead.n_i),
                                        dtype=np.float64)
                fObj[skew_key].create_dataset('taucalc_local', data=taucalc_local)


    if verbose:
        print(f"--- {rank_idstr} : ...initialization complete ! --- ")
    
    return


def taucalc(OTFSkewers_i, skewCosmoCalc, density_scale, comm, precision=np.float64, verbose=False):
    '''
    Calculate the effective optical depth for each skewer along an axis

    Args:
        OTFSkewers_i (ChollaOnTheFlySkewers_i): holds all skewer info along an axis
        skewCosmoCalc (ChollaSkewerCosmoCalculator): holds optical depth function
        density_scale (float) : scalar to change the HI number density
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

        skewerID_arr = np.arange(OTFSkewers_i.OTFSkewersiHead.n_skews)
        skewerIDs_rank = np.argwhere((skewerID_arr % size) == rank).flatten()

        # loop over each skewer
        for nSkewerID in skewerIDs_rank:

            # grab skewer data & calculate effective optical depth
            vel = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get('los_velocity')[nSkewerID, :]
            densityHI = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get('HI_density')[nSkewerID, :]
            temp = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get('temperature')[nSkewerID, :]

            densityHI_scaled = densityHI * density_scale
            taus = skewCosmoCalc.optical_depth_Hydrogen(densityHI_scaled, vel, temp)
            fluxes = np.exp(- taus)
            meanF = np.mean(fluxes)

            # update bool arr, and tau arrs
            fObj[skew_key]['taucalc_eff_match'][nSkewerID] = -1. * np.log(meanF)
            fObj[skew_key]['taucalc_local_match'][nSkewerID] = taus


    if verbose:
        print(f"--- {rank_idstr} : Effective optical depth calculation completed along ", OTFSkewers_i.OTFSkewersiHead.skew_key)

    return



def main():
    '''
    Append the array of optical depth of mean flux for a skewer file
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
        print(f"--- {rank_idstr} : Target optical depth is {args.tau_target:.8e} ---")

    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()

    if args.verbose:
        print(f"--- {rank_idstr} : skewer file {skewer_fPath} is a real file ! ---")

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath, comm)

    # define total number of new density scalars to try
    nmax = 50

    # add progress attribute, boolean mask for whether tau is calculated, and tau itself
    t_init_start = MPI.Wtime()
    init_taucalc(OTFSkewers, args.tau_target, nmax, comm, restart=args.restart, verbose=args.verbose)
    t_init_end = MPI.Wtime()
    if args.verbose:
        print(f"--- {rank_idstr} : Took {t_init_end - t_init_start:.4e} secs to initialize info ---")

    # create cosmology and snapshot header
    chCosmoHead = ChollaCosmologyHead(OTFSkewers.Omega_M, OTFSkewers.Omega_R, 
                                    OTFSkewers.Omega_K, OTFSkewers.Omega_L,
                                    OTFSkewers.w0, OTFSkewers.wa, OTFSkewers.H0)

    if args.verbose:
        print(f"--- {rank_idstr} : Cosmo Head and OTF Skewer objects created ---")

    OTFSkewers_x = OTFSkewers.get_skewersx_obj()
    OTFSkewers_y = OTFSkewers.get_skewersy_obj()
    OTFSkewers_z = OTFSkewers.get_skewersz_obj()

    if args.verbose:
        print(f"--- {rank_idstr} : OTFSkewers in each axis are created ---")

    skewCosmoCalc_x = ChollaSkewerCosmoCalculator(OTFSkewers.current_a, chCosmoHead, OTFSkewers.nx, OTFSkewers.dx_h_kpc, precision)
    skewCosmoCalc_y = ChollaSkewerCosmoCalculator(OTFSkewers.current_a, chCosmoHead, OTFSkewers.ny, OTFSkewers.dy_h_kpc, precision)
    skewCosmoCalc_z = ChollaSkewerCosmoCalculator(OTFSkewers.current_a, chCosmoHead, OTFSkewers.nz, OTFSkewers.dz_h_kpc, precision)

    if args.verbose:
        print(f"--- {rank_idstr} : Skewer Cosmo Calculator objects created ---")

    
    # initialize scalar to change the HI number density
    density_scale = 0.

    # initialize optical depth to compare against target
    curr_taucalc_eff_match = 0.

    # define the relative and absolute tolerance to match optical depths
    rtol, atol = 1.e-6, 1.e-6
    converged = False

    # grab density scale ranges
    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r', driver='mpio', comm=comm) as fObj:
        density_scale_delta_max = fObj.attrs['density_scale_delta_max'].item()
        density_scale_delta_min = fObj.attrs['density_scale_delta_min'].item()

    if args.verbose and rank == 0:
        print(f"--- {rank_idstr} : We are aiming for relative/absolute tol of {rtol:.4e} / {atol:.4e}")

    # calculate current effective optical depth for first guess
    if rank == 0:
        if args.verbose:
            print(f"--- {rank_idstr} : Calculating current effective optical depth ... ---")
        nskewers_x = OTFSkewers_x.OTFSkewersiHead.n_skews
        nskewers_y = OTFSkewers_y.OTFSkewersiHead.n_skews
        nskewers_z = OTFSkewers_z.OTFSkewersiHead.n_skews
        nskewers_tot = nskewers_x + nskewers_y + nskewers_z

        # make arrays of all effective optical depths along each axis
        tau_eff_key = "taucalc_eff"
        tau_eff_x = np.zeros(nskewers_x, dtype=precision)
        tau_eff_y = np.zeros(nskewers_y, dtype=precision)
        tau_eff_z = np.zeros(nskewers_z, dtype=precision)
        tau_eff_x[ : ] = OTFSkewers_x.get_skeweralldata(tau_eff_key, dtype=precision)
        tau_eff_y[ : ] = OTFSkewers_y.get_skeweralldata(tau_eff_key, dtype=precision)
        tau_eff_z[ : ] = OTFSkewers_z.get_skeweralldata(tau_eff_key, dtype=precision)

        # group all effective optical depths
        tau_eff_all = np.zeros(nskewers_tot, dtype=precision)
        tau_eff_all[ : (nskewers_x)] = tau_eff_x
        tau_eff_all[ (nskewers_x) : (nskewers_x + nskewers_y)] = tau_eff_y
        tau_eff_all[ (nskewers_x + nskewers_y) : ] = tau_eff_z

        # calculate fluxes
        fluxes_eff_all = np.exp(-1. * tau_eff_all)

        # calculate mean of effective flux and its optical depth
        meanF_eff = np.mean(fluxes_eff_all)
        medF_eff = np.percentile(fluxes_eff_all, 50)
        curr_taucalc_eff_match = -1. * np.log(medF_eff)

        if args.verbose:
            print(f"--- {rank_idstr} : Current effective optical depth is {curr_taucalc_eff_match:.8e}---")

        # store current values
        density_scale_history = np.zeros(nmax+1)
        taucalc_eff_match_history = np.zeros(nmax+1)
        density_scale_history[0] = 1.
        taucalc_eff_match_history[0] = curr_taucalc_eff_match

        # are we providing an overall boost to number density or decrease
        overall_boost = args.tau_target > curr_taucalc_eff_match
        
        # calculate initial scalar for HIdensity and create array to save
        density_scale_delta = np.abs(1. - (args.tau_target / curr_taucalc_eff_match))
        
        if overall_boost:
            density_scale = 1. + (density_scale_delta)
            if args.verbose:
                density_scale_min = 1. + density_scale_delta_min
                density_scale_max = 1. + density_scale_delta_max
                print(f"--- {rank_idstr} : we are boosting the density within the range of {density_scale_min:.4e} and {density_scale_max:.4e}")
        else:
            density_scale = 1. / (1. + density_scale_delta)
            if args.verbose:
                density_scale_min = 1. / (1. + density_scale_delta_max)
                density_scale_max = 1. / (1. + density_scale_delta_min)
                print(f"--- {rank_idstr} : we are decreasing the density within the range of {density_scale_min:.4e} and {density_scale_max:.4e}")

        taucalc_eff_atol = np.abs(args.tau_target - curr_taucalc_eff_match)
        taucalc_eff_rtol = taucalc_eff_atol / args.tau_target
        converged = (taucalc_eff_atol < atol) and (taucalc_eff_rtol < rtol)

    comm.Barrier()
    density_scale = comm.bcast(density_scale, root=0)
    converged = comm.bcast(converged, root=0)

    t_start = MPI.Wtime()
    n = 0
    while (not converged and n < nmax):
        if args.verbose:
            print(f"--- {rank_idstr} : Starting optical depth calculation with density scale of {density_scale:.4f} ---")

        t_start_iter = MPI.Wtime()

        taucalc(OTFSkewers_x, skewCosmoCalc_x, density_scale, comm, precision, args.verbose)
        if args.verbose:
            t_x = MPI.Wtime()
            print(f"--- {rank_idstr} : Took {t_x - t_start_iter:.4e} secs to calculate tau along x ---")

        taucalc(OTFSkewers_y, skewCosmoCalc_y, density_scale, comm, precision, args.verbose)
        if args.verbose:
            t_y = MPI.Wtime()
            print(f"--- {rank_idstr} : Took {t_y - t_x:.4e} secs to calculate tau along y ---")

        taucalc(OTFSkewers_z, skewCosmoCalc_z, density_scale, comm, precision, args.verbose)
        if args.verbose:
            t_z = MPI.Wtime()
            print(f"--- {rank_idstr} : Took {t_z - t_y:.4e} secs to calculate tau along z ---")
            print(f"--- {rank_idstr} : Calling barrier to collect all taus ---")

        # ensure all optical depths are calculated
        comm.Barrier()
        if rank == 0:
            if args.verbose:
                print(f"--- {rank_idstr} : Starting new effective optical depth calculation ---")

            tau_eff_match_key = 'taucalc_eff_match'
            tau_eff_x = np.zeros(nskewers_x, dtype=precision)
            tau_eff_y = np.zeros(nskewers_y, dtype=precision)
            tau_eff_z = np.zeros(nskewers_z, dtype=precision)
            tau_eff_x[ : ] = OTFSkewers_x.get_skeweralldata(tau_eff_match_key, dtype=precision)
            tau_eff_y[ : ] = OTFSkewers_y.get_skeweralldata(tau_eff_match_key, dtype=precision)
            tau_eff_z[ : ] = OTFSkewers_z.get_skeweralldata(tau_eff_match_key, dtype=precision)

            # group all effective optical depths
            tau_eff_all = np.zeros(nskewers_tot, dtype=precision)
            tau_eff_all[ : (nskewers_x)] = tau_eff_x
            tau_eff_all[ (nskewers_x) : (nskewers_x + nskewers_y)] = tau_eff_y
            tau_eff_all[ (nskewers_x + nskewers_y) : ] = tau_eff_z

            # calculate fluxes
            fluxes_eff_all = np.exp(-1. * tau_eff_all)

            # calculate mean of effective flux and its optical depth
            meanF_eff = np.mean(fluxes_eff_all)
            medF_eff = np.percentile(fluxes_eff_all, 50)
            curr_taucalc_eff_match = -1. * np.log(medF_eff)

            density_scale_history[n+1] = density_scale
            taucalc_eff_match_history[n+1] = curr_taucalc_eff_match

            taucalc_eff_atol = np.abs(args.tau_target - curr_taucalc_eff_match)
            taucalc_eff_rtol = taucalc_eff_atol / args.tau_target
            converged = (taucalc_eff_atol < atol) and (taucalc_eff_rtol < rtol)


            if args.verbose:
                print(f"--- {rank_idstr} : On iteration {n:.0f}, with density scale {density_scale:.4f} we find...")
                print(f"... optical depth : {curr_taucalc_eff_match:.8e} ")
                print(f"... rtol : {taucalc_eff_rtol:.4e} ")
                print(f"... atol : {taucalc_eff_atol:.4e} ")
                print(f"--- {rank_idstr} : density scale history is {density_scale_history[:n+1]}")
                print(f"--- {rank_idstr} : taucalc_eff_match_history {taucalc_eff_match_history[:n+1]}")

            if (overall_boost): # increased too much
                density_scale_delta2large = args.tau_target < curr_taucalc_eff_match
            else: # decreased too much
                density_scale_delta2large = args.tau_target > curr_taucalc_eff_match

            if np.isnan(curr_taucalc_eff_match):
                density_scale_delta2large = True

            if density_scale_delta2large:
                # we have hit the max we can span the delta
                density_scale_delta_max = density_scale_delta
            else: 
                # we can drive the lower bound of the delta at least to where it's at
                density_scale_delta_min = density_scale_delta

            # new density scale delta is midway of min/max in log-space
            l_density_scale_delta_min = np.log10(density_scale_delta_min)
            l_density_scale_delta_max = np.log10(density_scale_delta_max)
            l_density_scale_delta_mid = (l_density_scale_delta_min + l_density_scale_delta_max) / 2.0
            density_scale_delta = 10**( l_density_scale_delta_mid )

            if not converged:
                if overall_boost:
                    density_scale = 1. + (density_scale_delta)
                    if args.verbose:
                        density_scale_min = 1. + density_scale_delta_min
                        density_scale_max = 1. + density_scale_delta_max
                else:
                    density_scale = 1. / (1. + density_scale_delta)
                    if args.verbose:
                        density_scale_min = 1. / (1. + density_scale_delta_max)
                        density_scale_max = 1. / (1. + density_scale_delta_min)

            if args.verbose:
                print(f"--- {rank_idstr} : lDelta min/max : {l_density_scale_delta_min:.4e} / {l_density_scale_delta_max:.4e} ---")
                print(f"--- {rank_idstr} : New density scale delta of : {density_scale_delta:.4e} ---")
                print(f"--- {rank_idstr} : New density scale range of {density_scale_min:.4e} / {density_scale_max:.4e} --- ")
                print(f"--- {rank_idstr} : New density scale of : {density_scale:.4e} ---")

            # iterate n
            n += 1
        
        comm.Barrier()

        # update density scales within skewers fpath
        density_scale_delta_max = comm.bcast(density_scale_delta_max, root=0)
        density_scale_delta_min = comm.bcast(density_scale_delta_min, root=0)
        with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+', driver='mpio', comm=comm) as fObj:
            fObj.attrs['density_scale_delta_max'] = density_scale_delta_max
            fObj.attrs['density_scale_delta_min'] = density_scale_delta_min

        # from rank0 get convergence criterion evaluation, iteration value, and new density scale
        converged = comm.bcast(converged, root=0)
        n = comm.bcast(n, root=0)
        density_scale = comm.bcast(density_scale, root=0)


    t_end = MPI.Wtime()


    fName_split = OTFSkewers.OTFSkewersfPath.name.split('.')
    fName_match = fName_split[0] + "_taumatch." + fName_split[1]
    tauMatch_fPath = OTFSkewers.OTFSkewersfPath.parent / Path(fName_match)


    if args.verbose:
        print(f"--- {rank_idstr} : Copying local and effective tau info to {tauMatch_fPath} ---")
    with h5py.File(tauMatch_fPath, 'r+', driver='mpio', comm=comm) as fObj_match:

        with h5py.File(OTFSkewers.OTFSkewersfPath, 'r', driver='mpio', comm=comm) as fObj:
            
            OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                        OTFSkewers.get_skewersy_obj(),
                        OTFSkewers.get_skewersz_obj()]

            for i, OTFSkewers_i in enumerate(OTFSkewers_lst):
                skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key
                skewerID_arr = np.arange(OTFSkewers_i.OTFSkewersiHead.n_skews)
                skewerIDs_rank = np.argwhere((skewerID_arr % size) == rank).flatten()

                for nSkewerID in skewerIDs_rank:
                    fObj_match[skew_key]['taucalc_eff'][nSkewerID] = fObj[skew_key]['taucalc_eff_match'][nSkewerID]
                    fObj_match[skew_key]['taucalc_local'][nSkewerID] = fObj[skew_key]['taucalc_local_match'][nSkewerID]

        if args.verbose:
            print(f"--- {rank_idstr} : Took {t_end - t_start:.4e} secs for entire calculation ---")

        fObj_match[f'calctime_{size:.0f}_nprocs_match'][rank] = t_end - t_start
        fObj_match[f'inittime_{size:.0f}_nprocs_match'][rank] = t_init_end - t_init_start
        if rank == 0:
            fObj_match['taucalc_eff_history'][:] = taucalc_eff_match_history[:]
            fObj_match['density_scale_history'][:] = density_scale_history[:]
    comm.Barrier()

    if args.verbose:
        print(f"--- {rank_idstr} : Done moving data over to taumatch file ---")
        print(f"--- {rank_idstr} : Deleting info from original skewers file  ---")

    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+', driver='mpio', comm=comm) as fObj:
        OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                    OTFSkewers.get_skewersy_obj(),
                    OTFSkewers.get_skewersz_obj()]
        del fObj.attrs['density_scale_delta_max']
        del fObj.attrs['density_scale_delta_min']
        del fObj.attrs['tau_target_match']

        for i, OTFSkewers_i in enumerate(OTFSkewers_lst):
            skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key
            del fObj[skew_key]['taucalc_eff_match']
            del fObj[skew_key]['taucalc_local_match']

    if args.verbose:
        print(f"--- {rank_idstr} : yayy, we're done ! ---")

if __name__=="__main__":
    main()

