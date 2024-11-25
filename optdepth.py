import argparse
import os

import numpy as np
import h5py
from scipy.special import erf



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

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument('-l', '--local', help='Whether to store local optical depths',
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser

###
# Create all data structures to fully explain optical depth calculation
# These data structures are pretty thorough, and not every line is readily needed
# but I prioritize readability over less lines of code
###

###
# Calculations+bookkeeping related to cosmology, snapshot, and how optical depth is calculated
###
# ChollaSnapHead                --> will hold scale factor
# ChollaCosmologyHead           --> cosmology-specific info
# ChollaSnapCosmologyHead       --> combines ChollaSnap+ChollaCosmo
# ChollaCosmoCalculator         --> calculator for cosmology snapshot (unit conversions)
# ChollaHydroCalculator         --> cgs constants & doppler param method (indpt of scale factor) 
# ChollaSkewerCosmoCalculator   --> implements optical depth calculation along skewer length

class ChollaSnapHead:
    '''
    Cholla Snapshot Head object
        Holds snapshot specific information
        Initialized with:
        - nSnap (int): number of the snapshot within run
        - scale_factor (float): scale factor at current snapshot
    '''

    def __init__(self, nSnap, scale_factor):
        self.nSnap = nSnap
        self.a = scale_factor


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

        # Normalization factors from Cholla source code method Initialize_Cosmology. Not 100% sure what they mean
        self.r0_gas = 1.0 # simulation ran with gas in [h-1 kpc] (???????)
        self.t0_gas = self.t_H0_cosmo / self.h_cosmo
        self.v0_gas = self.r0_gas / self.t0_gas
        self.rho0_gas = self.rho_crit0_cosmo * self.OmegaM
        self.phi0_gas = self.v0_gas * self.v0_gas  # energy units
        self.e0_gas = self.v0_gas * self.v0_gas
        self.p0_gas = self.rho0_gas * self.v0_gas * self.v0_gas # pressure units

        # conversion factors between cosmo [kpc/km Msun kyr] and cgs [cm gram sec] units
        # these factors DO NOT account for comoving units (ie, scale factor not incorporated)
        # multiplying array (in cgs units) by array_cgs2cosmo provides array in cosmo units
        # multiplying array (in cosmo units) by array_cosmo2cgs provides array in cgs units
        self.density_cgs2cosmo = self.rho_crit0_cosmo # [h2 Msun kpc-3]
        self.density_cosmo2cgs = 1. / self.density_cgs2cosmo

        self.velocity_cgs2cosmo = self.km_cgs # [km s-1]
        self.velocity_cosmo2cgs = 1. / self.velocity_cgs2cosmo

        self.mom_cgs2cosmo = self.density_cgs2cosmo * self.km_cgs # [h2 Msun kpc-3  km s-1]
        self.mom_cosmo2cgs = 1. / self.mom_cgs2cosmo


class ChollaSnapCosmologyHead:
    '''
    Cholla Snapshot Cosmology header object
        Serves as a header holding information that combines a ChollaCosmologyHead
            with a specific scale factor with the snapshot header object.
        
        Initialized with:
            snapHead (ChollaSnapHead): provides current redshift
            cosmoHead (ChollaCosmologyHead): provides helpful information of cosmology & units

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, snapHead, cosmoHead):
        self.a = snapHead.a
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

    def dvHubble(self, dx):
        '''
        Return the Hubble flow through a cell

        Args:
            dx (float): comoving distance between cells (kpc)
        Returns:
            (float): Hubble flow over a cell (km/s)
        '''
        # convert [kpc] to [h-1 kpc]
        dx_h = dx / self.cosmoHead.h_cosmo

        dxh_cgs = dx_h * self.cosmoHead.kpc_cgs # h^-1 kpc * (#cm / kpc) =  h^-1 cm
        dxh_Mpc = dxh_cgs / self.cosmoHead.Mpc_cgs # h^-1 cm / (#cm / Mpc) = h^-1 Mpc

        # convert to physical length
        dxh_Mpc_phys = dxh_Mpc * self.a

        return self.Hubble() * dxh_Mpc_phys


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
            snapHead (ChollaSnapHead): provides current redshift
            cosmoHead (ChollaCosmologyHead): provides helpful information of cosmology & units
            n_los (int): number of cells along line-of-sight
            dx (float): comoving distance between cells (kpc)
            dtype (np type): (optional) numpy precision to initialize output arrays
        
        Objects including ghost cells are suffixed with _ghost

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, snapHead, cosmoHead, n_los, dx, dtype=np.float32):
        self.n_los = n_los
        self.n_ghost = int(0.1 * n_los) # take 10% from bruno
        self.dx = dx

        # number of line-of-sight cells including ghost cells
        self.n_los_ghost = self.n_los + 2 * self.n_ghost

        # create ChollaCosmoCalc object
        self.snapCosmoHead = ChollaSnapCosmologyHead(snapHead, cosmoHead)
        calc_dims, calc_dims_ghost = (self.n_los,), (self.n_los_ghost,)
        self.snapCosmoCalc = ChollaCosmoCalculator(self.snapCosmoHead, calc_dims, dtype=dtype)
        self.snapCosmoCalc_ghost = ChollaCosmoCalculator(self.snapCosmoHead, calc_dims_ghost, dtype=dtype)

        # create HydroCalc objects
        self.hydroCalc = ChollaHydroCalculator(calc_dims, dtype=dtype)
        self.hydroCalc_ghost = ChollaHydroCalculator(calc_dims_ghost, dtype=dtype)

        # calculate Hubble flow through one cell
        dvHubble = self.snapCosmoHead.dvHubble(self.dx) # [km s-1]
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

    def optical_depth_Hydrogen(self, densityHI, velocity_pec, temp, use_forloop=True):
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

        if use_forloop:
            # OLD IMPLEMENTATION
            for losid in range(self.n_los_ghost):
                vH_L, vH_R = self.vHubbleL_ghost_cgs[losid], self.vHubbleR_ghost_cgs[losid]
                # calculate line center shift in terms of broadening scale
                y_L = (vH_L - velocity_phys_ghost_cgs) / doppler_param_ghost_cgs
                y_R = (vH_R - velocity_phys_ghost_cgs) / doppler_param_ghost_cgs
                # [cm3 * # density] = [cm3 * cm-3] = []
                tau_ghost[losid] = sigma_Lya * np.sum(nHI_phys_ghost_cgs * (erf(y_R) - erf(y_L))) / 2.0
        else:
            # NEW IMPLEMENTATION w/o for-loop, uses more memory
            vHL_repeat = np.repeat(self.vHubbleL_ghost_cgs, self.n_los_ghost).reshape((self.n_los_ghost, self.n_los_ghost))
            vHR_repeat = np.repeat(self.vHubbleR_ghost_cgs, self.n_los_ghost).reshape((self.n_los_ghost, self.n_los_ghost))

            density_repeat = np.repeat(nHI_phys_ghost_cgs, self.n_los_ghost).reshape((self.n_los_ghost, self.n_los_ghost)).T
            vel_repeat = np.repeat(velocity_phys_ghost_cgs, self.n_los_ghost).reshape((self.n_los_ghost, self.n_los_ghost)).T
            doppler_repeat = np.repeat(doppler_param_ghost_cgs, self.n_los_ghost).reshape((self.n_los_ghost, self.n_los_ghost)).T

            yL_all = (vHL_repeat - vel_repeat) / doppler_repeat
            yR_all = (vHR_repeat - vel_repeat) / doppler_repeat

            tau_ghost[:] = sigma_Lya * np.sum(density_repeat * (erf(yR_all) - erf(yL_all)), axis=1) / 2.0


        # clip edges
        tau = tau_ghost[self.n_ghost : -self.n_ghost]

        return tau


# Skewer-specific information that interacts with skewers for a given skewer file
# ChollaOnTheFlySkewerHead      --> Holds skewer id
# ChollaOnTheFlySkewer          --> Grabs data from file
# ChollaOnTheFlySkewers_iHead   --> Holds skewer group
# ChollaOnTheFlySkewers_i       --> Creates ChollaOnTheFlySkewer object
# ChollaOnTheFlySkewers         --> Creates ChollaOnTheFlySkewers_i object

class ChollaOnTheFlySkewerHead:
    '''
    Cholla On The Fly Skewer Head

    Holds information regarding a specific individual skewer

        Initialized with:
        - skew_id (int): id of the skewer
        - n_i (int): length of the skewer
        - skew_key (str): string to access skewer

    '''
    def __init__(self, skew_id, n_i, skew_key):
        self.skew_id = skew_id
        self.n_i = n_i
        self.skew_key = skew_key


class ChollaOnTheFlySkewer:
    '''
    Cholla On The Fly Skewer
    
    Holds skewer specific information to an output with methods to 
            access data for that output

        Initialized with:
        - ChollaOTFSkewerHead (ChollaOnTheFlySkewerHead): header
            information associated with skewer
        - fPath (str): file path to skewers output

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, ChollaOTFSkewerHead, fPath):
        self.OTFSkewerHead = ChollaOTFSkewerHead
        self.fPath = fPath

        self.HI_str = 'HI_density'
        self.HeII_str = 'HeII_density'
        self.density_str = 'density'
        self.vel_str = 'los_velocity'
        self.temp_str = 'temperature'

        self.allkeys = {self.HI_str, self.HeII_str, self.density_str,
                        self.vel_str, self.temp_str}

    def check_datakey(self, data_key):
        '''
        Check if a requested data key is valid to be accessed in skewers file

        Args:
            data_key (str): key string that will be used to access hdf5 dataset
        Return:
            (bool): whether data_key is a part of expected data keys
        '''

        return data_key in self.allkeys

    def get_skewerdata(self, key, dtype=np.float32):
        '''
        Return a specific skewer dataset

        Args:
            key (str): key to access data from hdf5 file
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): requested dataset
        '''

        assert self.check_datakey(key)

        arr = np.zeros(self.OTFSkewerHead.n_i, dtype=dtype)
        fObj = h5py.File(self.fPath, 'r')
        arr[:] = fObj[self.OTFSkewerHead.skew_key].get(key)[self.OTFSkewerHead.skew_id, :]
        fObj.close()

        return arr

    def get_HIdensity(self, dtype=np.float32):
        '''
        Return the HI density array

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): HI density
        '''

        return self.get_skewerdata(self.HI_str, dtype=dtype)

    def get_losvelocity(self, dtype=np.float32):
        '''
        Return the line-of-sight velocity array

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): line-of-sight velocity
        '''

        return self.get_skewerdata(self.vel_str, dtype=dtype)

    def get_temperature(self, dtype=np.float32):
        '''
        Return the temperature array

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): temperature
        '''

        return self.get_skewerdata(self.temp_str, dtype=dtype)


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
        - fPath (str): file path to skewers output

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, ChollaOTFSkewersiHead, fPath):
        self.OTFSkewersiHead = ChollaOTFSkewersiHead
        self.fPath = fPath

        self.HI_str = 'HI_density'
        self.HeII_str = 'HeII_density'
        self.density_str = 'density'
        self.vel_str = 'los_velocity'
        self.temp_str = 'temperature'

        self.allkeys = {self.HI_str, self.HeII_str, self.density_str,
                        self.vel_str, self.temp_str}

    def get_skewer_obj(self, skewid):
        '''
        Return ChollaOnTheFlySkewer object of this analysis

        Args:
            skew_id (int): skewer id
        Return:
            OTFSkewer (ChollaOnTheFlySkewer): skewer object
        '''
        OTFSkewerHead = ChollaOnTheFlySkewerHead(skewid, self.OTFSkewersiHead.n_i,
                                                 self.OTFSkewersiHead.skew_key)

        return ChollaOnTheFlySkewer(OTFSkewerHead, self.fPath)


class ChollaOnTheFlySkewers:
    '''
    Cholla On The Fly Skewers
    
    Holds on-the-fly skewers specific information to an output with methods to 
            create specific skewer objects

        Initialized with:
        - nSkewer (nSkewer): number of the skewer output
        - SkewersPath (str): directory path to skewer output files
        - ChollaGrid (ChollaGrid): grid holding domain information

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, nSkewer, SkewersPath):
        self.OTFSkewersfPath = SkewersPath + '/' + str(nSkewer) + '_skewers.h5'

        self.xskew_str = "skewers_x"
        self.yskew_str = "skewers_y"
        self.zskew_str = "skewers_z"

        # set grid information (ncells, dist between cells, nstride)
        self.set_gridinfo()

        # set cosmology params
        self.set_cosmoinfo()

        # grab current hubble param & info needed to calculate hubble flow
        H = self.get_currH()
        cosmoh = self.H0 / 100.

        # calculate proper distance along each direction
        dxproper = self.dx * self.current_a / cosmoh
        dyproper = self.dy * self.current_a / cosmoh
        dzproper = self.dz * self.current_a / cosmoh

        # calculate hubble flow through a cell along each axis
        self.dvHubble_x = H * dxproper
        self.dvHubble_y = H * dyproper
        self.dvHubble_z = H * dzproper

    def set_gridinfo(self, datalength_str='density'):
        '''
        Set grid information by looking at attribute of file object and shape of 
            data sets
        
        Args:
            - datalength_str (str): (optional) key to dataset used to find the
                number of skewers and cells along an axis
        Returns:
            ...
        '''

        fObj = h5py.File(self.OTFSkewersfPath, 'r')

        # grab length of box in units of [kpc]
        Lx, Ly, Lz = np.array(fObj.attrs['Lbox'])

        # set number of skewers and stride number along each direction 
        nskewersx, self.nx = fObj[self.xskew_str][datalength_str].shape
        nskewersy, self.ny = fObj[self.yskew_str][datalength_str].shape
        nskewersz, self.nz = fObj[self.zskew_str][datalength_str].shape

        fObj.close()

        # we know nskewers_i = (nj * nk) / (nstride_i * nstride_i)
        # so nstride_i = sqrt( (nj * nk) / (nskewers_i) )
        self.nstride_x = int(np.sqrt( (self.ny * self.nz)/(nskewersx) ))
        self.nstride_y = int(np.sqrt( (self.nz * self.nx)/(nskewersy) ))
        self.nstride_z = int(np.sqrt( (self.nx * self.ny)/(nskewersz) ))

        # save cell distance in each direction to later calculate hubble flow
        self.dx = Lx / self.nx
        self.dy = Ly / self.ny
        self.dz = Lz / self.nz

        # convert kpc --> Mpc
        self.dx /= 1.e3
        self.dy /= 1.e3
        self.dz /= 1.e3

    def set_cosmoinfo(self):
        '''
        Set cosmological attributes for this object

        Args:
            ...
        Returns:
            ...
        '''

        fObj = h5py.File(self.OTFSkewersfPath, 'r')

        self.Omega_R = fObj.attrs['Omega_R'].item()
        self.Omega_M = fObj.attrs['Omega_M'].item()
        self.Omega_L = fObj.attrs['Omega_L'].item()
        self.Omega_K = fObj.attrs['Omega_K'].item()

        self.w0 = fObj.attrs['w0'].item()
        self.wa = fObj.attrs['wa'].item()

        self.H0 = fObj.attrs['H0'].item() # expected in km/s/Mpc
        self.current_a = fObj.attrs['current_a'].item()
        self.current_z = fObj.attrs['current_z'].item()

        fObj.close()

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

        OTFSkewerx = ChollaOnTheFlySkewers_i(OTFSkewersxHead, self.OTFSkewersfPath)

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

        OTFSkewery = ChollaOnTheFlySkewers_i(OTFSkewersyHead, self.OTFSkewersfPath)

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

        OTFSkewerz = ChollaOnTheFlySkewers_i(OTFSkewerszHead, self.OTFSkewersfPath)

        return OTFSkewerz


# create functions that will open and write data
def init_taucalc(OTFSkewers, verbose=False, local=False):
    '''
    Initialize the calculation of the effective optical depth. For each skewers_i axis
        group, create three things:
        1. (attribute) progress for optical depth
        2. (dataset) boolean mask whether optical depth has been calculated for 
            a specific skewer
        3. (dataset) median of local optical depths for a skewer
    
    Current implementation assumes the same nStride along each axis

    Args:
        OTFSkewers (ChollaOnTheFlySkewers): holds OTF skewers specific info
        verbose (bool): (optional) whether to print important information
        local (bool): (optional) whether to save local optical depths
    Returns:
        ...

    '''

    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+') as fObj:
        if verbose:
            print(f'\t...initializing optical depth calculations for file {OTFSkewers.OTFSkewersfPath}')

        OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                          OTFSkewers.get_skewersy_obj(),
                          OTFSkewers.get_skewersz_obj()]

        # add progress attribute, boolean mask for whether tau is calculated, and tau itself
        for i, OTFSkewers_i in enumerate(OTFSkewers_lst):
            if verbose:
                print(f"\t\t...initializing arrays and attributes along axis {i:.0f}")
            skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key

            taucalc_bool = np.zeros(OTFSkewers_i.OTFSkewersiHead.n_skews, dtype=bool)
            taucalc = np.zeros(OTFSkewers_i.OTFSkewersiHead.n_skews, dtype=np.float64)

            if 'taucalc_prog' not in dict(fObj[skew_key].attrs).keys():
                fObj[skew_key].attrs['taucalc_prog'] = 0.
            if 'taucalc_bool' not in fObj[skew_key].keys():
                fObj[skew_key].create_dataset('taucalc_bool', data=taucalc_bool)
            if 'taucalc_eff' not in fObj[skew_key].keys():
                fObj[skew_key].create_dataset('taucalc_eff', data=taucalc)
            if ((local) and 'taucalc_local' not in fObj[skew_key].keys()):
                taucalc_local = np.zeros((OTFSkewers_i.OTFSkewersiHead.n_skews, OTFSkewers_i.OTFSkewersiHead.n_i),
                                          dtype=np.float64)
                fObj[skew_key].create_dataset('taucalc_local', data=taucalc_local)

    if verbose:
        print("...initialization complete !")

    return


def taucalc(OTFSkewers_i, skewCosmoCalc, precision=np.float64, verbose=False, local=False):
    '''
    Calculate the effective optical depth for each skewer along an axis

    Args:
        OTFSkewers_i (ChollaOnTheFlySkewers_i): holds all skewer info along an axis
        skewCosmoCalc (ChollaSkewerCosmoCalculator): holds optical depth function
        precision (np type): (optional) numpy precision to use
        verbose (bool): (optional) whether to print important information
        local (bool): (optional) whether to save local optical depths
    Returns:
        ...
    '''

    skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key

    with h5py.File(OTFSkewers_i.fPath, 'r+') as fObj:
        curr_progress = fObj[skew_key].attrs['taucalc_prog']
        progress_tenperc = int(curr_progress // 0.1)
        if verbose:
            print(f"Starting calculations at {100 * curr_progress:.2f} % complete")

        # loop over each skewer
        for nSkewerID in range(OTFSkewers_i.OTFSkewersiHead.n_skews):
            # skip skewers whose optical depth already calculated
            if (fObj[skew_key]['taucalc_bool'][nSkewerID]):
                continue

            # grab skewer data & calculate effective optical depth
            OTFSkewer = OTFSkewers_i.get_skewer_obj(nSkewerID)
            vel = OTFSkewer.get_losvelocity(precision)
            densityHI = OTFSkewer.get_HIdensity(precision)
            temp = OTFSkewer.get_temperature(precision)
            taus = skewCosmoCalc.optical_depth_Hydrogen(densityHI, vel, temp, use_forloop=True)
            tau_eff = np.median(taus)

            # update attr, bool arr, and tau arr
            fObj[skew_key].attrs['taucalc_prog'] += (1. / OTFSkewers_i.OTFSkewersiHead.n_skews)
            fObj[skew_key]['taucalc_bool'][nSkewerID] = True
            fObj[skew_key]['taucalc_eff'][nSkewerID] = tau_eff
            if local:
                fObj[skew_key]['taucalc_local'][nSkewerID] = taus

            if ((verbose) and ( (fObj[skew_key].attrs['taucalc_prog'] // 0.1) > progress_tenperc) ):
                print(f"--- Completed {fObj[skew_key].attrs['taucalc_prog'] * 100 : .0f} % at skewer {nSkewerID:.0f} ---")
                progress_tenperc += 1

    if verbose:
        print("Effective optical depth calculation completed along ", OTFSkewers_i.OTFSkewersiHead.skew_key)

    return



def main():
    '''
    Append the array of median optical depths for a skewer file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer file : {args.skewfname} ---")
        if args.local:
            print(f"--- We are saving local optical depths (!) ---")
        else:
            print(f"--- We are NOT saving local optical depths (!) ---")

    precision = np.float64
    
    _ = '''
    # create set of all param values we are interested in saving
    params2grab = {'nx', 'ny', 'nz', 'xmin', 'ymin', 'zmin', 'xlen', 'ylen', 'zlen',
                    'H0', 'Omega_M', 'Omega_L', 'Omega_K', 'Omega_R', 'Omega_b', 'w0', 'wa',
                    'analysis_scale_outputs_file', 'skewersdir', 'lya_skewers_stride'}

    # read in params from param text file
    params = {}
    with open(args.param, 'r') as paramfile:
        for line in paramfile:
            # strip whitespace, then split by key-value pair assignment
            keyval_str = '='
            linesplit = line.strip().split(keyval_str)
            is_keyvalpair = len(linesplit) == 2
            if is_keyvalpair:
                key_str, val_str = linesplit
                if key_str in params2grab:
                    params[key_str] = val_str

    if len(params) != len(params2grab):
        print(f'--- MISSING FOLLOWING PARAMS IN PARAM TXT FILE {args.param} ---')
        for param in params2grab:
            if param not in params.keys():
                print('\t - ', param)
        print('--- PLEASE FIX... EXITING ---')
        exit()
    '''

    # convert relative path to skewer file name to absolute file path
    cwd = os.getcwd()
    if args.skewfname[0] != '/':
        relative_path = args.skewfname
        args.skewfname = cwd + '/' + relative_path

    # seperate the skewer output number and skewer directory
    skewfName = args.skewfname.split('/')[-1]
    nSkewerOutput = int(skewfName.split('_')[0])
    skewersdir = args.skewfname[:-(len(skewfName)+1)]

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(nSkewerOutput, skewersdir)

    # add progress attribute, boolean mask for whether tau is calculated, and tau itself
    init_taucalc(OTFSkewers, args.verbose, args.local)

    # create cosmology and snapshot header
    chCosmoHead = ChollaCosmologyHead(OTFSkewers.Omega_M, OTFSkewers.Omega_R, 
                                      OTFSkewers.Omega_K, OTFSkewers.Omega_L,
                                      OTFSkewers.w0, OTFSkewers.wa, OTFSkewers.H0)
    snapHead = ChollaSnapHead(nSkewerOutput + 1, OTFSkewers.current_a) # snapshots are index-1
        
    OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(), OTFSkewers.get_skewersy_obj(),
                      OTFSkewers.get_skewersz_obj()]

    # complete calculation
    for i, OTFSkewers_i in enumerate(OTFSkewers_lst):
        if args.verbose:
            print(f"Starting calculation along axis {i:.0f}")

        if (i == 0):
            nlos = OTFSkewers.nx
            dx = OTFSkewers.dx
        elif (i == 1):
            nlos = OTFSkewers.ny
            dx = OTFSkewers.dy
        elif (i == 2):
            nlos = OTFSkewers.nz
            dx = OTFSkewers.dz

        skewCosmoCalc = ChollaSkewerCosmoCalculator(snapHead, chCosmoHead, nlos, dx, precision)
        taucalc(OTFSkewers_i, skewCosmoCalc, precision, args.verbose, args.local)


        

if __name__=="__main__":
    main()

