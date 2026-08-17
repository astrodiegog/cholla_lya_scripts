"""
This script calculates the local optical depth for all skewers in a Cholla
skewer file. The default native datasets expected in the skewer files
are at least the ionized HI density, peculiar velocity, and temperature.
We assume a Gaussian line profile. This script assigns ranks to different
skewers within a skewer file to produce the computation more quickly. This
script assumes that HDF5 was built with parallelized version, and the python
package h5py was built with parallelized version as well.

Usage:
$ python3 optdepth_mp.py 0_skewers.h5 -v
"""

import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from time import time
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

    parser.add_argument("axis", help='axis along which skewers were drawn', type=int)

    parser.add_argument('nprocs', help='Number of processes to pool workers', type=int)

    parser.add_argument('-p', '--peculiarless', help='Remove peculiar velocities',
                        action='store_true')

    parser.add_argument('-r', '--restart', help='Reset progress bool array', 
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
# ChollaCosmologyHead			--> cosmology-specific info
# ChollaSnapCosmologyHead		--> combines ChollaSnap+ChollaCosmo
# ChollaCosmoCalculator			--> calculator for cosmology snapshot (unit conversions)
# ChollaHydroCalculator			--> cgs constants & doppler param method (indpt of scale factor) 
# ChollaSkewerCosmoCalculator	--> implements optical depth calculation along skewer length


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

    def pool_optical_depth_Hydrogen_nopec(self, data):
        '''
        Compute the optical depth for each cell along the line-of-sight w/o 
            peculiar velocity with a single data argument for pooling workers

        Args:
            data (arr) : array of shape (2, nLOS) where the first dimension
                describes the ionized Hydrogen comoving density [h2 Msun kpc-3]
                and the second describes the temperature [K]
        Returns:
            tau (arr): optical depth for each cell
        '''

        assert data.shape[0] == 2
        return self.optical_depth_Hydrogen_nopec(data[0,:], data[1,:])

    def optical_depth_Hydrogen_nopec(self, densityHI, temp):
        '''
        Compute the optical depth for each cell along the line-of-sight w/o 
            peculiar velocity

        Args:
            densityHI (arr): ionized Hydrogen comoving density [h2 Msun kpc-3]
            temp (arr): temperature [K]
        Returns:
            tau (arr): optical depth for each cell
        '''
        assert densityHI.size == self.n_los
        assert temp.size == self.n_los

        # convert comoving density to physical density then to cgs
        densityHI_phys = self.snapCosmoCalc.physical_density(densityHI)
        densityHI_phys_cgs = self.snapCosmoCalc.density_cosmo2cgs(densityHI_phys) # [g cm-3]

        # calculate column number density & extend to ghost cells
        nHI_phys_cgs = densityHI_phys_cgs / self.hydroCalc.mp # [cm-3]
        nHI_phys_ghost_cgs = self.extend_ghostcells(nHI_phys_cgs)

        # set physical velocity as Hubble flow
        velocity_phys_ghost_cgs = self.vHubbleC_ghost_cgs # [cm s-1]

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



    def pool_optical_depth_Hydrogen(self, data):
        '''
        Compute the optical depth for each cell along the line-of-sight with 
            a single data argument for pooling workers

        Args:
            data (arr) : array of shape (3, nLOS) where the first dimension
                describes the ionized Hydrogen comoving density [h2 Msun kpc-3],
                the second dimension holds the peculiar velocity [km s-1],
                and the third describes the temperature [K]
        Returns:
            (arr): optical depth for each cell
        '''

        assert data.shape[0] == 3
        return self.optical_depth_Hydrogen(data[0,:], data[1,:], data[2,:])


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
# ChollaSkewers			--> Creates ChollaSkewers_i object



class ChollaSkewers:
    '''
    Cholla On The Fly Skewers

    Holds on-the-fly skewers specific information to an output with methods to 
            create specific skewer objects

        Initialized with:
        - fPath (PosixPath): file path to skewers output	

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, fPath, axis):
        self.SkewersfPath = fPath.resolve() # convert to absolute path
        assert self.SkewersfPath.is_file() # make sure file exists


        self.axis = axis

        assert 0 <= self.axis and self.axis <= 2

        # set grid information (ncells, dist between cells)
        self.set_gridinfo()
        dlos_h_Mpc = self.dlos_h_kpc / 1.e3 # [h-1 Mpc]

        # set cosmology params
        self.set_cosmoinfo()

        # grab current hubble param & info needed to calculate hubble flow
        H = self.get_currH()  # [km s-1 Mpc-1]
        cosmoh = self.H0 / 100.

        # calculate proper distance along each direction
        dlosproper = dlos_h_Mpc * self.current_a / cosmoh # [Mpc]

        # calculate Hubble flow through a cell along each axis
        self.dvHubble_los = H * dlosproper # [km s-1]

    def set_gridinfo(self, datalength_str='HI_density'):
        '''
        Set grid information by looking at attribute of file object and shape of 
            data sets
        
        Args:
            datalength_str (str): (optional) key to dataset used to find the
                number of skewers and cells along an axis
        Returns:
            ...
        '''
        with h5py.File(self.SkewersfPath, 'r') as fObj:
            # save cell distance in LOS direction to later calculate hubble flow
            self.dlos_h_kpc = np.array(fObj.attrs['dx'])[self.axis]

            # set number of skewers along each direction 
            self.nSkewers, self.nlos = fObj[datalength_str].shape

        # save length of box in units of [h-1 kpc]
        self.L_los_h_kpc = self.dlos_h_kpc * self.nlos

        return

    def print_gridinfo(self):
        '''
        Print the grid information related to the skewers

        Args:
            ...
        Returns:
            ...
        '''


        print(f"---  We have {self.nSkewers} total number of skewers ---")
        print(f"--- Line of sight is {self.nlos} cells long with total length of {self.L_los_h_kpc:.4f} h-1 kpc---")
        print(f"--- Cell width along LOS are {self.dlos_h_kpc:.4f} h-1 kpc")
        print(f"--- Hubble flow along a cell is {self.dvHubble_los:.4f} km / s")


    def set_cosmoinfo(self):
        '''
        Set cosmological attributes for this object

        Args:
            ...
        Returns:
            ...
        '''

        with h5py.File(self.SkewersfPath, 'r') as fObj:
            #self.Omega_R = fObj.attrs['Omega_R'].item()
            self.Omega_M = fObj.attrs['Omega_M'].item()
            self.Omega_L = fObj.attrs['Omega_L'].item()
            #self.Omega_K = fObj.attrs['Omega_K'].item()

            #self.w0 = fObj.attrs['w0'].item()
            #self.wa = fObj.attrs['wa'].item()
            self.Omega_R, self.Omega_K = 0., 0.
            self.w0, self.wa = -1.0, 0.

            H0_kpc = fObj.attrs['H0'].item() # expected in km/s/kpc
            self.H0 = H0_kpc * 1.e3 # km / s / kpc --> km / s / Mpc
            self.current_a = fObj.attrs['Current_a'].item()
            self.current_z = fObj.attrs['Current_z'].item()
        
        return

    def print_cosmoinfo(self):
        '''
        Print the cosmology information related to this snapshot

        Args:
            ...
        Returns:
            ...
        '''

        print(f"--- studying skewers at z={self.current_z:.3f} / a={self.current_a:.3f} ---")
        print(f"--- energy density info ---")
        print(f"--- \t OmegaM = {self.Omega_M:.4f} ---")
        print(f"--- \t OmegaDE = {self.Omega_L:.4f} ---")
        print(f"--- \t OmegaR = {self.Omega_R:.4f} ---")
        print(f"--- \t OmegaK = {self.Omega_K:.4f} ---")
        print(f"--- \t w0, wa = {self.w0:.4f}, {self.wa:.4f} ---")
        print(f"--- H0 = {self.H0:.4f} km / s / Mpc")
        print(f"--- H(z={self.current_z}) = {self.get_currH():.4f} km / s / Mpc")


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


#####
# Script specific functions
#####


def init_taucalc(Skewers, restart = False, verbose=False, nopec=False):
    '''
    Initialize the calculation of the effective optical depth. For each skewers_i axis
        group, create three things:
        1. (attribute) progress for optical depth
        2. (dataset) boolean mask whether optical depth has been calculated for 
            a specific skewer
        3. (dataset) optical depth of mean flux for a skewer

    Current implementation assumes the same nStride along each axis

    Args:
        Skewers (ChollaSkewers): holds skewers specific info
        restart (bool): (optional) whether to reset progress and set all 
                        taucalc_bool to False
        verbose (bool): (optional) whether to print important information
        nopec (bool): (optional) whether we remove peculiar velocity or not
    Returns:
        ...
    '''


    if nopec:
        taucalc_bool_key = 'taucalc_nopec_bool'
        taucalc_eff_key = 'taucalc_nopec_eff'
        taucalc_local_key = 'taucalc_nopec_local'
        calctime_key = f'calctime_nopec_no_nprocs'
        inittime_key = f'inittime_nopec_no_nprocs'
    else:
        taucalc_bool_key = 'taucalc_bool'
        taucalc_eff_key = 'taucalc_eff'
        taucalc_local_key = 'taucalc_local'
        calctime_key = f'calctime_no_nprocs'
        inittime_key = f'inittime_no_nprocs'

    with h5py.File(Skewers.SkewersfPath, 'r+') as fObj:

        if verbose:
            print(f'--- \t...initializing optical depth calculations for file {Skewers.SkewersfPath} ---')

        if calctime_key not in fObj.keys():
            calctime_arr = np.zeros(1, dtype=np.float64)
            fObj.create_dataset(calctime_key, data=calctime_arr)
        elif restart:
            fObj[calctime_key][:] = 0.

        if inittime_key not in fObj.keys():
            calctime_arr = np.zeros(1, dtype=np.float64)
            fObj.create_dataset(inittime_key, data=calctime_arr)
        elif restart:
            fObj[inittime_key][:] = 0.

        # add progress attribute, boolean mask for whether tau is calculated, and tau itself
        if verbose:
            print(f"--- \t\t...initializing arrays and attributes ---")

        taucalc_bool = np.zeros(Skewers.nSkewers, dtype=bool)
        taucalc_eff = np.zeros(Skewers.nSkewers, dtype=np.float64)

        taucalc_local = np.zeros((Skewers.nSkewers, Skewers.nlos),
                                        dtype=np.float64)

        if taucalc_bool_key not in fObj.keys():
            fObj.create_dataset(taucalc_bool_key, data=taucalc_bool)
        elif restart:
            fObj[taucalc_bool_key][:] = False

        if taucalc_eff_key not in fObj.keys():
            fObj.create_dataset(taucalc_eff_key, data=taucalc_eff)

        if taucalc_local_key not in fObj.keys():
            fObj.create_dataset(taucalc_local_key, data=taucalc_local)


    if verbose:
        print(f"--- ...initialization complete ! --- ")

    return


def taucalc(Skewers, skewCosmoCalc, precision=np.float64, verbose=False, nopec=False, nprocs=2):
    '''
    Calculate the effective optical depth for each skewer along an axis

    Args:
        Skewers (ChollaSkewers): holds all skewer info along an axis
        skewCosmoCalc (ChollaSkewerCosmoCalculator): holds optical depth function
        precision (np type): (optional) numpy precision to use
        verbose (bool): (optional) whether to print important information
        nopec (bool): (optional) whether we remove peculiar velocity or not
        nprocs (int): (optional) how many processes to use in pooling workers
    Returns:
        ...
    '''


    if nopec:
        taucalc_bool_key = 'taucalc_nopec_bool'
        taucalc_eff_key = 'taucalc_nopec_eff'
        taucalc_local_key = 'taucalc_nopec_local'
    else:
        taucalc_bool_key = 'taucalc_bool'
        taucalc_eff_key = 'taucalc_eff'
        taucalc_local_key = 'taucalc_local'
    with h5py.File(Skewers.SkewersfPath, 'r+') as fObj:
        taucalc_bool = fObj[taucalc_bool_key]
        curr_progress = np.sum(taucalc_bool) / taucalc_bool.size
        if verbose:
            print(f"--- Starting calculations at {100 * curr_progress:.2f} % complete along ---")

        # only load skewers whose optical depth have not been calculated
        skewers_2load_mask = ~taucalc_bool[:]
        skewerIDs_2load = np.argwhere(skewers_2load_mask).flatten()
        nSkewers_2load = skewerIDs_2load.size

        if nopec:
            full_data = np.zeros( (nSkewers_2load, 2, Skewers.nlos) , dtype=precision)
            full_data[ :, 0, : ] = fObj.get('HI_density')
            full_data[ :, 1, : ] = fObj.get('temperature')
        else:
            full_data = np.zeros( (nSkewers_2load, 3, Skewers.nlos) , dtype=precision)
            full_data[ :, 0, : ] = fObj.get('HI_density')
            full_data[ :, 1, : ] = fObj.get('velocity')
            full_data[ :, 2, : ] = fObj.get('temperature')

          
        with Pool(processes=nprocs) as pool:
            chunksize = 25 # chosen from experimentation
            if nopec:
                res = pool.imap(skewCosmoCalc.pool_optical_depth_Hydrogen_nopec, full_data, chunksize=chunksize)
            else:
                res = pool.imap(skewCosmoCalc.pool_optical_depth_Hydrogen, full_data, chunksize=chunksize)

            for skewerID_local, taus_skew in enumerate(res):
                fluxes = np.exp(-1. * taus_skew)
                meanF = np.mean(fluxes)
                skewerID = skewerIDs_2load[skewerID_local]
                fObj[taucalc_bool_key][skewerID] = True
                fObj[taucalc_eff_key][skewerID] = -1. * np.log(meanF)
                fObj[taucalc_local_key][skewerID] = taus_skew

            
    if verbose:
        print(f"--- Effective optical depth calculation completed ---")

    return



def main():
    '''
    Append the array of optical depth of mean flux for a skewer file
    '''


    # Create parser 
    parser = create_parser()
    # Save args
    args = parser.parse_args()
    if args.verbose:
        print(f"--- Args parsed and created ! ---")

    if args.verbose:
        print(f"--- Using {args.nprocs:.0f} processes!")
        if args.peculiarless:
            print(f'--- Not including peculiar velocities ! ---')
        else:
            print(f'--- Including peculiar velocities ! ---')

    precision = np.float64

    assert args.nprocs > 1

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()

    assert 0 <= args.axis and args.axis <= 2

    if args.verbose:
        print(f"--- skewer file {skewer_fPath} is a real file ! ---")
        ix_str = ['x', 'y', 'z']
        print(f"--- skewers drawn along {ix_str[args.axis]} axis ---")

    # create ChollaSkewers object
    Skewers = ChollaSkewers(skewer_fPath, args.axis)

    if args.verbose:
        print(f"--- Skewer object created ---")
        Skewers.print_cosmoinfo()
        Skewers.print_gridinfo()

    # add progress attribute, boolean mask for whether tau is calculated, and tau itself
    t_init_start = time()
    init_taucalc(Skewers, restart=args.restart, verbose=args.verbose, nopec=args.peculiarless)
    t_init_end = time()
    if args.verbose:
        print(f"--- Took {t_init_end - t_init_start:.4e} secs to initialize info ---")

    # create cosmology and snapshot header
    chCosmoHead = ChollaCosmologyHead(Skewers.Omega_M, Skewers.Omega_R, 
                                    Skewers.Omega_K, Skewers.Omega_L,
                                    Skewers.w0, Skewers.wa, Skewers.H0)


    if args.verbose:
        print(f"--- Cosmo Head object created ---")

    skewCosmoCalc = ChollaSkewerCosmoCalculator(Skewers.current_a, chCosmoHead, 
                                                  Skewers.nlos, Skewers.dlos_h_kpc, precision)

    if args.verbose:
        print(f"--- Skewer Cosmo Calculator object created ---")

    t_start = time()
    taucalc(Skewers, skewCosmoCalc, precision, args.verbose, args.peculiarless, args.nprocs)
    t_end =time()

    with h5py.File(Skewers.SkewersfPath, 'r+') as fObj:
        if args.verbose:
            print(f"--- Took {t_end - t_start:.4e} secs for entire calculation ---")

        if args.peculiarless:
            fObj[f'calctime_nopec_no_nprocs'] = t_end - t_start
            fObj[f'inittime_nopec_no_nprocs'] = t_init_end - t_init_start
        else:
            fObj[f'calctime_no_nprocs'][:] = t_end - t_start
            fObj[f'inittime_no_nprocs'][0] = t_init_end - t_init_start
    

if __name__=="__main__":
    main()

