#!/usr/bin/env python3
"""
This script groups flux power spectra from each x-y-z axis. This script loops
    over every group in nOutput_fluxpowerspectrum_optdepthbin.h5 and combines
    power spectra along each axis by making a common k-mode space that takes
    logarithmic steps as an input. Given some differential step in logarithmic 
    k-space dlogk, we take the largest Hubble flow along all axes in order to 
    drive to the smallest k value. We start with the smallest k value and 
    create a k-mode array that takes logarithmic steps like dlogk. We then loop
    over every k_x, k_y, and k_z value to find where they land on this new 
    k-mode array, and use it to add the flux power spectrum from some axis, 
    while normalizing by the number of number of k values that land there.

Usage for 0.002 logarithmic k-mode steps:
    $ python3 group_powspec_dlogk.py 0_fluxpowerspectrum_optdepthbin.h5 0.002 -v
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.special import erf
import h5py



###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the FPS optdepthbin name.
        Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Group power spectra by differential log step size")

    parser.add_argument("FPS_optdepthbin_fname", help='Optical depth binned Flux Power Spectra file', type=str)

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


###
# Create all data structures to fully explain power spectrum grouping
###


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




def main():
    '''
    Group the flux power spectrum from each axis
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()


    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at file : {args.FPS_optdepthbin_fname} ---")

        print(f"--- We are placing power spectra in dlogk : {args.dlogk} ---")



    precision = np.float64

    analysis_fPath = Path(args.FPS_optdepthbin_fname).resolve()
    assert analysis_fPath.is_file()

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    if args.verbose:
        print("--- Grabbing required information to rebin power spectra ---")

    # grab required info
    with h5py.File(analysis_fPath, 'r') as fObj:
        # grab cosmo info
        Omega_M = fObj.attrs.get('Omega_M')
        Omega_R = fObj.attrs.get('Omega_R')
        Omega_K = fObj.attrs.get('Omega_K')
        Omega_L = fObj.attrs.get('Omega_L')
        H0, w0, wa = fObj.attrs.get('H0'), fObj.attrs.get('w0'), fObj.attrs.get('wa')

        # grab snapshot info
        scale_factor = fObj.attrs.get('scale_factor')
        redshift = fObj.attrs.get('redshift')

        # grab grid info
        Lbox = fObj.attrs.get('Lbox')[:]
        nCells = fObj.attrs.get('nCells')[:]

        # grab FFT info
        k_x = fObj.get('k_x')[:]
        k_y = fObj.get('k_y')[:]
        k_z = fObj.get('k_z')[:]

    # calculate number of nFFT bins
    nFFTs = np.zeros_like(nCells, dtype=np.int64)
    nFFTs[:] = 1 + nCells / 2

    # create cosmology objects                
    chCosmoHead = ChollaCosmologyHead(Omega_M, Omega_R, Omega_K, Omega_L, w0, wa, H0)
    chSnapCosmoHead = ChollaSnapCosmologyHead(scale_factor, chCosmoHead)
    dx, dy, dz = Lbox / nCells

    # calculate Hubble flow through cell & across entire box
    dvHubble_x = chSnapCosmoHead.dvHubble(dx)
    dvHubble_y = chSnapCosmoHead.dvHubble(dy)
    dvHubble_z = chSnapCosmoHead.dvHubble(dz)
    u_max_x = dvHubble_x * nCells[0]
    u_max_y = dvHubble_y * nCells[1]
    u_max_z = dvHubble_z * nCells[2]

    # find largest u_max period & associated smallest period
    u_FFT_min, u_FFT_max = 0., 0.
    if u_max_x > u_FFT_max:
        u_FFT_max = u_max_x
        u_FFT_min = u_max_x / (nFFTs[0] - 1.)
    if u_max_y > u_FFT_max:
        u_FFT_max = u_max_y
        u_FFT_min = u_max_y / (nFFTs[1] - 1.)
    if u_max_z > u_FFT_max:
        u_FFT_max = u_max_z
        u_FFT_min = u_max_z / (nFFTs[2] - 1.)
    if args.verbose:
        print(f"--- The largest Hubble flow across box is : {u_FFT_max:.4e} km s-1 ---")
        print(f"--- Normalized by number of FFT bins, the smallest sampled velocity is : {u_FFT_min:.4e} km s-1 ---")

    # create most inclusive kmin/kmax    
    l_kmin = np.log10( (2. * np.pi) / u_FFT_max) + np.log10(0.99) # add log(0.99) to match Cholla
    l_kmax = np.log10( (2. * np.pi) / u_FFT_min)

    if args.verbose:
        print(f"--- log_10(kmin) : {l_kmin:.4e} ---")
        print(f"--- log_10(kmax) : {l_kmax:.4e} ---")

    # create k value edges for inclusive power spectrum
    n_bins = int(1 + ((l_kmax - l_kmin) / args.dlogk))
    iter_arr = np.arange(n_bins + 1)
    k_edges = np.zeros(n_bins+1, dtype=precision)
    k_edges[:] = 10**(l_kmin + (args.dlogk * iter_arr))  

    if args.verbose:
        print(f"--- We have {n_bins:.0f} number of bins we are placing FPS into---")

    # for each axis, calculate where each k_i value lands on k_edges
    # then find out _how many_ fft modes land in each kedge bin
    fft_binids_float_x = (np.log10(k_x) - l_kmin ) / args.dlogk
    fft_binids_x = np.zeros(nFFTs[0], dtype=np.int64)
    fft_binids_x[:] = np.floor(fft_binids_float_x)
    fft_nbins_kedges_x = np.zeros(n_bins, dtype=precision)
    for fft_bin_id in fft_binids_x[1:]:
        fft_nbins_kedges_x[fft_bin_id] += 1.
        
    fft_binids_float_y = (np.log10(k_y) - l_kmin ) / args.dlogk
    fft_binids_y = np.zeros(nFFTs[1], dtype=np.int64)
    fft_binids_y[:] = np.floor(fft_binids_float_y)
    fft_nbins_kedges_y = np.zeros(n_bins, dtype=precision)
    for fft_bin_id in fft_binids_y[1:]:
        fft_nbins_kedges_y[fft_bin_id] += 1.

    fft_binids_float_z = (np.log10(k_z) - l_kmin ) / args.dlogk
    fft_binids_z = np.zeros(nFFTs[2], dtype=np.int64)
    fft_binids_z[:] = np.floor(fft_binids_float_z)
    fft_nbins_kedges_z = np.zeros(n_bins, dtype=precision)
    for fft_bin_id in fft_binids_z[1:]:
        fft_nbins_kedges_z[fft_bin_id] += 1.

    # (protect against dividing by zero)
    fft_nbins_kedges_x[fft_nbins_kedges_x == 0] = 1.
    fft_nbins_kedges_y[fft_nbins_kedges_y == 0] = 1.
    fft_nbins_kedges_z[fft_nbins_kedges_z == 0] = 1.

    if args.verbose:
        print("--- Just figured out how many FFT modes land in each bin. Now writing data ---")

    # create grouped FPS for each quantile
    with h5py.File(analysis_fPath, 'r+') as fObj:
        nQuantiles = fObj.attrs.get('nquantiles')
        nRanges = fObj.attrs.get('nranges')

        # flush out old dlogk analysis
        dlogk_analysis_alive = 'dlogk' in fObj.attrs
        if dlogk_analysis_alive:
            _ = fObj.attrs.pop('dlogk')
            del fObj['k_edges_dlogk']

        # write attr
        _ = fObj.attrs.create('dlogk', args.dlogk)

        # write kedges
        _ = fObj.create_dataset('k_edges_dlogk', data=k_edges)

        for nQuantile in range(nQuantiles):
            currQuantile_key = f'FluxPowerSpectrum_quantile_{nQuantile:.0f}'
            quantile_group = fObj[currQuantile_key]
            # grab FPS arrays
            FPS_x = quantile_group.get('FPS_x')[:]
            FPS_y = quantile_group.get('FPS_y')[:]
            FPS_z = quantile_group.get('FPS_z')[:]

            # create FPS array
            FPS_currQuantile = np.zeros(n_bins, dtype=precision)
            
            # add contribution from each FPS axis, avg by number FFT bins in kedge
            _ = np.add.at(FPS_currQuantile, fft_binids_x[1:], FPS_x[1:])
            FPS_currQuantile /= fft_nbins_kedges_x

            _ = np.add.at(FPS_currQuantile, fft_binids_y[1:], FPS_y[1:])
            FPS_currQuantile /= fft_nbins_kedges_y

            _ = np.add.at(FPS_currQuantile, fft_binids_z[1:], FPS_z[1:])
            FPS_currQuantile /= fft_nbins_kedges_z

            # avg by 3 for each axis
            FPS_currQuantile /= 3.

            # delete data set if it is alive
            if dlogk_analysis_alive:
                del quantile_group['FPS_dlogk']
        
            # write data
            _ = quantile_group.create_dataset('FPS_dlogk', data=FPS_currQuantile)
    
        for nRange in range(nRanges):
            currRange_key = f'FluxPowerSpectrum_range_{nRange:.0f}'
            range_group = fObj[currRange_key]

            # grab FPS arrays
            FPS_x = range_group.get('FPS_x')[:]
            FPS_y = range_group.get('FPS_y')[:]
            FPS_z = range_group.get('FPS_z')[:]

            # create FPS array
            FPS_currRange = np.zeros(n_bins, dtype=precision)
            
            # add contribution from each FPS axis, avg by number FFT bins in kedge
            _ = np.add.at(FPS_currRange, fft_binids_x[1:], FPS_x[1:])
            FPS_currRange /= fft_nbins_kedges_x
            
            _ = np.add.at(FPS_currRange, fft_binids_y[1:], FPS_y[1:])
            FPS_currRange /= fft_nbins_kedges_y

            _ = np.add.at(FPS_currRange, fft_binids_z[1:], FPS_z[1:])
            FPS_currRange /= fft_nbins_kedges_z

            # avg by 3 for each axis
            FPS_currRange /= 3.

            # delete data set if it is alive
            if dlogk_analysis_alive:
                del range_group['FPS_dlogk']



    if args.verbose:
        print("--- Done ! ---")
 


if __name__=="__main__":
    main()


