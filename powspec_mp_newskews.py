#!/usr/bin/env python3
"""
This script will compute the transmitted flux power spectrum from the new 
    skewer files outputs. This script assumes that we have the taucalc_local
    local optical depth and taucalc_eff effective optical depth (optical depth 
    corresponding to the mean flux along a skewer). This script will create a
    file with dataset

fluxpowerspectrum.h5
├── attrs
├── effopticaldepth [nskewers]
└── fluxpowerspectra [nskewers, nkmodes]

Usage for skewers is
    $ python3 powspec_newskews.py $skewerDir -v

"""

import argparse
from pathlib import Path

from multiprocessing import Pool
import numpy as np
import h5py


###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the skewer dierectory.
         Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Compute power spectra in range bin")

    parser.add_argument("skewDirName", help='Cholla skewer output directory name', type=str)

    parser.add_argument("axis", help='axis along which skewers were drawn', type=int)

    parser.add_argument('nprocs', help='Number of processes to pool workers', type=int)

    parser.add_argument('-o', '--outdir', help='Output directory for flux power spectrum file', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


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
            #self.dlos_h_kpc = np.array(fObj.attrs['dx'])[self.axis]
            self.dx_h_kpc = np.array(fObj.attrs['dx'])
            self.L_box_h_kpc = np.array(fObj.attrs['domain'])

            # set number of skewers along each direction 
            self.nSkewers, self.nlos = fObj[datalength_str].shape

        # save number of cells along each direction
        self.nCells = np.array(self.L_box_h_kpc / self.dx_h_kpc, dtype=np.uint64)

        # save length of box in units of [h-1 kpc]
        self.L_los_h_kpc = self.L_box_h_kpc[self.axis]
        #self.L_los_h_kpc = self.dlos_h_kpc * self.nlos

        # save number of cells along LOS
        self.dlos_h_kpc = self.dx_h_kpc[self.axis]

        return

    def print_gridinfo(self):
        '''
        Print the grid information related to the skewers

        Args:
            ...
        Returns:
            ...
        '''


        print(f"--- We have {self.nSkewers} total number of skewers in one file ---")
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

        print(f"--- Studying skewers at z={self.current_z:.3f} / a={self.current_a:.3f} ---")
        print(f"--- Energy density info ---")
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


###
# Calculations related to the geometry along an axis for a power spectrum calculation
###
# ChollaFluxPowerSpectrumHead    --> hold nfft and methods to get related k-mode arrays

class ChollaFluxPowerSpectrumHead:
    '''
    Cholla Flux Power Spectrum Head
    
    Holds information regarding the power spectrum calculation

        Initialized with:
        - nlos (int): number of line-of-sight cells
        - dvHubble (float): differential Hubble flow velocity across a cell
        - global_flux_mean (float): global mean transmitted HI flux
            to later scale local flux deviations

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, nlos, dvHubble, global_flux_mean):
        self.n_los = nlos
        self.n_fft = int(self.n_los / 2 + 1)
        self.dvHubble = dvHubble

        assert global_flux_mean >= 0
        self.global_flux_mean = global_flux_mean

        # calculate Hubble flow across entire box (max velocity)
        self.u_max = self.dvHubble * self.n_los

        self.l_kmin = np.log10( (2. * np.pi) / (self.u_max) )
        self.l_kmax = np.log10( (2. * np.pi * (self.n_fft - 1.) ) / (self.u_max) )


    def get_kvals_fft(self, dtype=np.float32):
        '''
        Return k-modes from the Fourier Transform

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            kcenters_fft (arr): k mode centers array
        '''

        kcenters_fft = np.zeros(self.n_fft, dtype=dtype)
        iter_arr = np.arange(self.n_fft, dtype=dtype)

        kcenters_fft[:] = (2. * np.pi * iter_arr) / (self.u_max)

        return kcenters_fft

    def get_flux_overdensity(self, local_opticaldepth, precision=np.float64):
        '''
        Return the flux overdensity given the local optical depths

        Args:
            local_opticaldepth (arr): local optical depth for single skewer of shape (nlos,)
        Returns:
            dFlux_skew (arr): flux fluctuations
        '''

        dFlux_skew = np.zeros_like(local_opticaldepth, dtype=precision)
        local_fluxes = np.exp(-1. * local_opticaldepth)

        dFlux_skew = (local_fluxes - self.global_flux_mean) / self.global_flux_mean

        return dFlux_skew

    def get_flux_power_spectra(self, local_opticaldepth, precision=np.float64):
        '''
        Return the flux power spectra along a single skewer

        Args:
            local_opticaldepth (arr): local optical depth for single skewer of shape (nlos,)
        Returns:
            PF_k (arr): flux fluctuation power spectra
        '''

        dFlux_skew = self.get_flux_overdensity(local_opticaldepth, precision=precision)

        # perform fft & calculate amplitude of fft
        fft = np.fft.rfft(dFlux_skew)
        fft2 = (fft.imag * fft.imag) + (fft.real * fft.real)

        # take avg & scale by umax
        delta_F_avg = fft2 / self.n_los / self.n_los
        P_k = self.u_max * delta_F_avg

        return P_k

    def get_FPS(self, local_opticaldepths, flux_mean_global=None, precision=np.float64):
        '''
        Return the Flux Power Spectrum given the local optical depths.
            Expect 2-D array of shape (number skewers, line-of-sight cells)

        Args:
            local_opticaldepths (arr): local optical depths of all skewers
            flux_mean_global (float): (optional) global mean flux to scale local deviations
            precision (np type): (optional) numpy precision to use
        Return:
            kmode_fft (arr): Fourier Transform k mode array
            P_k_mean (arr): mean transmitted flux power spectrum within kmode edges
        '''
        assert local_opticaldepths.ndim == 2
        assert local_opticaldepths.shape[1] == self.n_los

        n_skews = local_opticaldepths.shape[0]

        # calculate local transmitted flux (& its mean)
        fluxes = np.exp(-local_opticaldepths)
        if flux_mean_global:
            assert flux_mean_global > 0 
            flux_mean = flux_mean_global
        else:
            flux_mean = np.mean(fluxes)

        # initialize total power array & delta F avg arrays
        delta_F_avg = np.zeros(self.n_fft , dtype=precision)
        P_k_tot = np.zeros(self.n_fft, dtype=precision)

        for nSkewerID in range(n_skews):
            # calculate flux fluctuation 
            dFlux_skew = (fluxes[nSkewerID] - flux_mean) / flux_mean

            # perform fft & calculate amplitude of fft
            fft = np.fft.rfft(dFlux_skew)
            fft2 = (fft.imag * fft.imag) + (fft.real * fft.real)

            # take avg & scale by umax
            delta_F_avg = fft2 / self.n_los / self.n_los
            P_k = self.u_max * delta_F_avg
            P_k_tot += P_k

        # average out by the number of skewers
        P_k_mean = P_k_tot / n_skews

        # grab k-mode values
        kmode_fft = self.get_kvals_fft(precision)

        return (kmode_fft, P_k_mean)


def main():
    '''
    Compute flux power spectrum and create fluxpowerspectrum.h5 file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer directory : {args.skewDirName} ---")
        

    if args.verbose:
        print(f"--- Using {args.nprocs:.0f} processes!")

    assert args.nprocs > 1

    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_dirPath = Path(args.skewDirName).resolve()
    assert skewer_dirPath.is_dir()

    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        outdir_dirPath = skewer_dirPath.resolve()

    if args.verbose:
        print(f"--- Placing output files in : {outdir_dirPath} ---")

    # get analysis file name
    outfile_fname = f"fluxpowerspectrum.h5"
    outfile_fPath = outdir_dirPath / Path(outfile_fname)
    outfile_exists = outfile_fPath.is_file()
    assert not outfile_exists

    if args.verbose:
        print(f'--- Saving file at : {outfile_fPath} ---')


    # make sure required keys are there
    tau_local_key = "taucalc_local"
    tau_eff_key = "taucalc_eff"
    req_keys = [tau_local_key, tau_eff_key]


    if args.verbose:
        print(f'--- Grabbing global info related to skewer and cosmo calcs ---')
    # assume all skewer files have the same geometry (grid, Lbox),
    # cosmology, and number of skewers
    numSkewFiles = 0
    numSkews_per_file = 0
    for i, skewfPath in enumerate(skewer_dirPath.iterdir()):
        numSkewFiles += 1
        if not i: # create base info
            Skewers = ChollaSkewers(skewfPath, args.axis)
            chCosmoHead = ChollaCosmologyHead(Skewers.Omega_M, Skewers.Omega_R, 
                                    Skewers.Omega_K, Skewers.Omega_L,
                                    Skewers.w0, Skewers.wa, Skewers.H0)

            #chFPSHead = ChollaFluxPowerSpectrumHead(Skewers.nlos, Skewers.dvHubble_los)
            numSkews_per_file = Skewers.nSkewers

    nSkews_tot = int(numSkewFiles * numSkews_per_file)
    if args.verbose:
        print(f"--- We have {numSkewFiles:.0f} total files, each with {numSkews_per_file:.0f} skewers ---")
        Skewers.print_cosmoinfo()
        Skewers.print_gridinfo()

    if args.verbose:
        print(f"--- Saving effective and local optical depths ---")
    tau_eff_tot = np.zeros(nSkews_tot, dtype=precision)
    tau_loc_tot = np.zeros((nSkews_tot, Skewers.nlos), dtype=precision)
    for i, skewfPath in enumerate(skewer_dirPath.iterdir()):
        with h5py.File(skewfPath, 'r') as fObj:
            tau_eff_tot[ i * numSkews_per_file : (i+1) * numSkews_per_file ] = fObj.get(tau_eff_key)
            tau_loc_tot[ i * numSkews_per_file : (i+1) * numSkews_per_file , :] = fObj.get(tau_local_key)

    if args.verbose:
        print(f"--- Calculating statistics of optical depth / flux ---")
    fluxes_eff = np.exp(-1. * tau_eff_tot)
    mean_flux_eff = np.mean(fluxes_eff)
    low_flux_eff = np.percentile(fluxes_eff, 16)
    med_flux_eff = np.percentile(fluxes_eff, 50)
    hi_flux_eff = np.percentile(fluxes_eff, 84)

    tau_mean_flux_eff = np.log(mean_flux_eff)
    tau_low_flux_eff = np.log(low_flux_eff)
    tau_med_flux_eff = np.log(med_flux_eff)
    tau_hi_flux_eff = np.log(hi_flux_eff)

    mean_tau_eff = np.mean(tau_eff_tot)
    low_tau_eff = np.percentile(tau_eff_tot, 16)
    med_tau_eff = np.percentile(tau_eff_tot, 50)
    hi_tau_eff = np.percentile(tau_eff_tot, 84)

    chFPSHead = ChollaFluxPowerSpectrumHead(Skewers.nlos, Skewers.dvHubble_los, 
                                            med_flux_eff)
    
    if args.verbose:
        print(f"--- Calculating flux power spectrum... ---")
    PF_k_tot = np.zeros( (nSkews_tot, chFPSHead.n_fft), dtype=precision)


    with Pool(processes=args.nprocs) as pool:
        chunksize = 25
        res = pool.imap(chFPSHead.get_flux_power_spectra, tau_loc_tot, chunksize=chunksize) 

        for skewerID, PF_k_skew in enumerate(res):
            PF_k_tot[skewerID] = PF_k_skew


    kvals = chFPSHead.get_kvals_fft(precision)
    PF_k_mean = np.mean(PF_k_tot, axis=0)
    PF_k_std = np.std(PF_k_tot, axis=0)
    PF_k_low = np.percentile(PF_k_tot, 16, axis=0)
    PF_k_med = np.percentile(PF_k_tot, 50, axis=0)
    PF_k_hi = np.percentile(PF_k_tot, 84, axis=0)


    if args.verbose:
        print(f"--- Writing info to {outfile_fPath} ---")

    with h5py.File(outfile_fPath, 'w') as fObj:
        _ = fObj.attrs.create('skewDir', str(skewer_dirPath))

        # start with cosmo info
        _ = fObj.attrs.create('Omega_R', chCosmoHead.OmegaR)
        _ = fObj.attrs.create('Omega_M', chCosmoHead.OmegaM)
        _ = fObj.attrs.create('Omega_L', chCosmoHead.OmegaL)
        _ = fObj.attrs.create('Omega_K', chCosmoHead.OmegaK)
        #_ = fObj.attrs.create('Omega_b', chCosmoHead.Omegab)
        _ = fObj.attrs.create('w0', chCosmoHead.w0)
        _ = fObj.attrs.create('wa', chCosmoHead.wa)
        _ = fObj.attrs.create('H0', chCosmoHead.H0)

        # sim info
        _ = fObj.attrs.create('Lbox', Skewers.L_box_h_kpc)
        _ = fObj.attrs.create('nCells', Skewers.nCells)
        _ = fObj.attrs.create('Llos', Skewers.L_los_h_kpc)
        _ = fObj.attrs.create('nlos', Skewers.nlos)
        _ = fObj.attrs.create('nSkewers', Skewers.nSkewers)

        # snapshot info
        _ = fObj.attrs.create('redshift', Skewers.current_z)
        _ = fObj.attrs.create('scale_factor', Skewers.current_a)

        # analysis info
        _ = fObj.create_dataset('k_modes', data=kvals)
        _ = fObj.attrs.create('dvHubble', chFPSHead.dvHubble)

        # optical depth / flux info
        _ = fObj.create_dataset('tau_eff', data=tau_eff_tot)
        _ = fObj.attrs.create('mean_flux_eff', mean_flux_eff)
        _ = fObj.attrs.create('low_flux_eff', low_flux_eff)
        _ = fObj.attrs.create('med_flux_eff', med_flux_eff)
        _ = fObj.attrs.create('hi_flux_eff', hi_flux_eff)
        _ = fObj.attrs.create('tau_mean_flux_eff', tau_mean_flux_eff)
        _ = fObj.attrs.create('tau_low_flux_eff', tau_low_flux_eff)
        _ = fObj.attrs.create('tau_med_flux_eff', tau_med_flux_eff)
        _ = fObj.attrs.create('tau_hi_flux_eff', tau_hi_flux_eff)
        _ = fObj.attrs.create('mean_tau_eff', mean_tau_eff)
        _ = fObj.attrs.create('low_tau_eff', low_tau_eff)
        _ = fObj.attrs.create('med_tau_eff', med_tau_eff)
        _ = fObj.attrs.create('hi_tau_eff', hi_tau_eff)

        # flux power spectrum info
        _ = fObj.create_dataset('PF_k_tot', data=PF_k_tot)
        _ = fObj.create_dataset('PF_k_mean', data=PF_k_mean)
        _ = fObj.create_dataset('PF_k_std', data=PF_k_std)
        _ = fObj.create_dataset('PF_k_low', data=PF_k_low)
        _ = fObj.create_dataset('PF_k_med', data=PF_k_med)
        _ = fObj.create_dataset('PF_k_hi', data=PF_k_hi)


if __name__=="__main__":
    main()



