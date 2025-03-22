#!/usr/bin/env python3
"""
This script takes the transmitted flux power spectrum binned by some number
    of quantiles bounded by a lower and upper effective optical depth. This
    script assumes that the optical depth has been computed and saved to the 
    skewer files as taucalc_local and taucalc_eff. This script will create a
    file with the following structure

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

    where the attributes for the root group will inherit the same attributes
    as from the skewer file. The attributes for each quantile group saves the 
    min, max, and mean effective optical depth of this quantile. The indices
    dataset describes the details of the skewer that lands in this quantile.
    Values will run from zero to the total number of skewers summed along all 
    axes, so we choose to have indices (0, nCells[0]) correspond to skewer IDs 
    along the x-axis, (nCells[0], nCells[1]) to skewer IDs along the y-axis, 
    and (nCells[0] + nCells[1], ) to skewer IDs along the z-axis. The flux
    power spectra in each direction is also saved in the quantiles.

Usage for 10 quantiles bounded by optical depths 0.001 and 100.0:
    $ python3 powspec_skewer_quantiles.py 0_skewers.h5 10 0.001 100.0 -v

    the output file will have the name 0_fluxpowerspectrum_optdepthbin.h5
"""

import argparse
from pathlib import Path

import numpy as np
import h5py



###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the skewer file name,
        number of quantiles, and bounding optical depth. Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Compute power spectra in nquantiles bins")

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument("optdepthlow", help='Lower effective optical depth limit to bin', type=float)

    parser.add_argument("optdepthupp", help='Upper effective optical depth limit to bin', type=float)

    parser.add_argument('-o', '--outdir', help='Output directory for analysis files', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


###
# Create all data structures to fully explain power spectrum calculation
# These data structures are pretty thorough, and not every line is readily needed
# but I prioritize readability over less lines of code
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

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, nlos, dvHubble):
        self.n_los = nlos
        self.n_fft = int(self.n_los / 2 + 1)
        self.dvHubble = dvHubble

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

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, ChollaOTFSkewersiHead, fPath):
        self.OTFSkewersiHead = ChollaOTFSkewersiHead
        self.fPath = fPath.resolve() # convert to absolute path
        assert self.fPath.is_file() # make sure file exists

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

    Values are returned in code units unless otherwise specified.
    '''

    def __init__(self, fPath):
        self.OTFSkewersfPath = fPath.resolve() # convert to absolute path
        assert self.OTFSkewersfPath.is_file() # make sure file exists

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
        H = self.get_currH() # [km s-1 Mpc-1]
        cosmoh = self.H0 / 100.

        # calculate proper distance along each direction
        dxproper = dx_Mpc * self.current_a / cosmoh # [h-1 Mpc]
        dyproper = dy_Mpc * self.current_a / cosmoh
        dzproper = dz_Mpc * self.current_a / cosmoh

        # calculate hubble flow through a cell along each axis
        self.dvHubble_x = H * dxproper # [km s-1]
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


        with h5py.File(self.OTFSkewersfPath, 'r') as fObj:
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

        # save cell distance in each direction to later calculate Hubble flow
        self.dx = Lx / self.nx
        self.dy = Ly / self.ny
        self.dz = Lz / self.nz
        
        return

    def set_cosmoinfo(self):
        '''
        Set cosmological attributes for this object

        Args:
            ...
        Returns:
            ...
        '''

        with h5py.File(self.OTFSkewersfPath, 'r') as fObj:
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


def main():
    '''
    Group skewers by effective optical depth, compute flux power spectrum, and
        create nOutput_fluxpowerspectrum_optdepthbin.h5 file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer file : {args.skewfname} ---")
        
        print(f"--- We have a range from {args.optdepthlow:.3e} to {args.optdepthupp:.3e}---")


    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()
    nOutput = int(skewer_fPath.name.split('_')[0])   # file names are nOutput_skewers.h5

    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        outdir_dirPath = skewer_fPath.parent.parent.resolve()

    if args.verbose:
        print(f"--- Placing output files in : {outdir_dirPath} ---")

    # get analysis file name
    outfile_fname = f"{nOutput:.0f}_fluxpowerspectrum_optdepthbin.h5"
    outfile_fPath = outdir_dirPath / Path(outfile_fname)
    outfile_exists = outfile_fPath.is_file()

    if args.verbose:
        print(f'--- Saving file at : {outfile_fPath} ---')
        if outfile_exists:
            print(f'--- File already exists, no need for global calculations ---')
        else:
            print(f'--- File does not exist, performing following global calculations: ')
            calc_string = f'--- Mean effective optical depth ---'
            calc_string += f'--- Mean local optical depth ---'
            calc_string += f'--- Mean effective flux + Mean effective flux optical depth ---'
            calc_string += f'--- Mean local flux + Mean local flux optical depth ---'
            print(calc_string)


    # ensure limits are reasonable
    assert args.optdepthlow >= 0
    assert args.optdepthupp > args.optdepthlow

    # make sure required keys are there
    tau_local_key = "taucalc_local"
    tau_eff_key = "taucalc_eff"
    req_keys = [tau_local_key, tau_eff_key]

    if args.verbose:
        print(f"--- Making sure {skewer_fPath} exists with required data ---")

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)
    OTFSkewers_x = OTFSkewers.get_skewersx_obj()
    OTFSkewers_y = OTFSkewers.get_skewersy_obj()
    OTFSkewers_z = OTFSkewers.get_skewersz_obj()

    for key in req_keys:
        assert OTFSkewers_x.check_datakey(key)
        assert OTFSkewers_y.check_datakey(key)
        assert OTFSkewers_z.check_datakey(key)

    precision = np.float64

    if args.verbose:
        print(f"--- Saving simulation box, skewer, and cosmology information ---")

    # save snapshot specific information
    scale_factor = OTFSkewers.current_a
    redshift = OTFSkewers.current_z

    # save cosmology info
    Omega_K, Omega_L = OTFSkewers.Omega_K, OTFSkewers.Omega_L
    Omega_M, Omega_R = OTFSkewers.Omega_M, OTFSkewers.Omega_R
    Omega_b, H0 = OTFSkewers.Omega_b, OTFSkewers.H0
    w0, wa = OTFSkewers.w0, OTFSkewers.wa
    # create length of box array (kpc)
    Lbox = np.zeros(3, dtype=precision)
    Lbox[0] = OTFSkewers.dx * OTFSkewers.nx
    Lbox[1] = OTFSkewers.dy * OTFSkewers.ny
    Lbox[2] = OTFSkewers.dz * OTFSkewers.nz
    # create number of cells array
    nCells = np.zeros(3, dtype=np.uint64)
    nCells[0] = int(OTFSkewers.nx)
    nCells[1] = int(OTFSkewers.ny)
    nCells[2] = int(OTFSkewers.nz)
    # create nFFT array
    nFFTs = np.zeros(3, dtype=np.uint64)
    nFFTs[0] = int(1. + nCells[0] / 2.)
    nFFTs[1] = int(1. + nCells[1] / 2.)
    nFFTs[2] = int(1. + nCells[2] / 2.)
    # create nstrides arrays
    nstrides = np.zeros(3, dtype=precision)
    nstrides[0] = OTFSkewers.nstride_x
    nstrides[1] = OTFSkewers.nstride_y
    nstrides[2] = OTFSkewers.nstride_z
    # calculate number of skewers
    nskewers_x = int((OTFSkewers.ny * OTFSkewers.nz) / (OTFSkewers.nstride_x * OTFSkewers.nstride_x))
    nskewers_y = int((OTFSkewers.nx * OTFSkewers.nz) / (OTFSkewers.nstride_y * OTFSkewers.nstride_y))
    nskewers_z = int((OTFSkewers.nx * OTFSkewers.ny) / (OTFSkewers.nstride_z * OTFSkewers.nstride_z))
    nskewers_tot = nskewers_x + nskewers_y + nskewers_z

    if args.verbose:
        print(f"--- We are sorting {nskewers_tot:.0f} total skewers ---")


    # make arrays of all effective optical depths along each axis
    # knowing the axis_skewid and nOutput, index will be [nOutput * nskewers_x + axis_skewid]   
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
    if not outfile_exists:
        tau_eff_all_mean = np.mean(tau_eff_all)

    # create a mask of all effective optical depths within our range
    tau_eff_all_inbounds_mask = (tau_eff_all > args.optdepthlow) & (tau_eff_all < args.optdepthupp)
    nskewers_inbounds = np.sum(tau_eff_all_inbounds_mask)
    if args.verbose:
        print(f"--- We have {nskewers_inbounds:.0f} skewers falling within effective optical depth range from total {nskewers_tot:.0f} skewers")
    assert nskewers_inbounds > 0


    # index of (0, nskewers_x) corresponds to x-axis
    # index of (nskewers_x, nskewers_x + nskewers_y) corresponds to y-axis
    # index of (nskewers_x + nskewers_y, nskewers_x + nskewers_y + nskewers_z) corresponds to z-axis
    indx_all_inbounds = np.argwhere(tau_eff_all_inbounds_mask).flatten()

    # make masks of the indices that fall within a specific axis
    indx_x_inbounds_mask = indx_all_inbounds < (nskewers_x)
    indx_y_inbounds_mask = ( (nskewers_x) < indx_all_inbounds) & (indx_all_inbounds < (nskewers_x + nskewers_y))
    indx_z_inbounds_mask = ( (nskewers_x + nskewers_y) < indx_all_inbounds)

    nskews_x_inbounds = np.sum(indx_x_inbounds_mask)
    nskews_y_inbounds = np.sum(indx_y_inbounds_mask)
    nskews_z_inbounds = np.sum(indx_z_inbounds_mask)

    # apply masks to get indices in each axis
    indx_x_inbounds = indx_all_inbounds[indx_x_inbounds_mask]
    indx_y_inbounds = indx_all_inbounds[indx_y_inbounds_mask]
    indx_z_inbounds = indx_all_inbounds[indx_z_inbounds_mask]

    # calculate the skewer id each index corresponds to
    skewid_x_inbounds = indx_x_inbounds % nskewers_x
    skewid_y_inbounds = indx_y_inbounds % nskewers_y
    skewid_z_inbounds = indx_z_inbounds % nskewers_z

    # save the local optical depths
    tau_local_x = OTFSkewers_x.get_skeweralldata(tau_local_key, dtype=precision)
    tau_local_y = OTFSkewers_y.get_skeweralldata(tau_local_key, dtype=precision)
    tau_local_z = OTFSkewers_z.get_skeweralldata(tau_local_key, dtype=precision)

    if not outfile_exists:
        # group local optical depths
        nCells_x = int(nskews_x * nCells[0])
        nCells_y = int(nskews_y * nCells[1])
        nCells_z = int(nskews_z * nCells[2])
        tau_local_all = np.zeros(tau_local_all, dtype=precision)
        tau_local_all[ : (nCells_x) ] = tau_local_x.flatten()
        tau_local_all[ (nCells_x) : (nCells_x + nCells_y)] = tau_local_y.flatten()
        tau_local_all[ (nCells_x + nCells_y) : ] = tau_local_z.flatten()
        tau_local_all_mean = np.mean(tau_local_all)

    # create an array for all local optical depths along an axis
    tau_local_x_inbounds = tau_local_x[skewid_x_inbounds].flatten()
    tau_local_y_inbounds = tau_local_y[skewid_y_inbounds].flatten()
    tau_local_z_inbounds = tau_local_z[skewid_z_inbounds].flatten()

    # calculate number of cells in bounds (same as summing all tau_local_inbounds sizes)
    nCells_x_inbounds = int(nskews_x_inbounds * nCells[0])
    nCells_y_inbounds = int(nskews_y_inbounds * nCells[1])
    nCells_z_inbounds = int(nskews_z_inbounds * nCells[2])
    nCells_inbounds = int(nCells_x_inbounds + nCells_y_inbounds + nCells_z_inbounds)
        
    # place all local optical depths
    tau_local_inbounds = np.zeros(nCells_inbounds, dtype=precision)
    tau_local_inbounds[ : (nCells_x_inbounds) ] = tau_local_x_inbounds
    tau_local_inbounds[ (nCells_x_inbounds) : (nCells_x_inbounds + nCells_y_inbounds) ] = tau_local_y_inbounds
    tau_local_inbounds[ (nCells_x_inbounds + nCells_y_inbounds) : ] = tau_local_z_inbounds
        
    # calculate fluxes and the mean
    fluxes_local_inbounds = np.exp(- tau_local_inbounds)
    fluxes_eff_inbounds = np.exp(- tau_eff_all[tau_eff_all_inbounds_mask])
    meanF_local_inbounds = np.mean(fluxes_local_inbounds)
    meanF_eff_inbounds = np.mean(fluxes_eff_inbounds)
    if not outfile_exists:
        fluxes_local_all = np.exp(- tau_local_all)
        fluxes_eff_all = np.exp(- tau_eff_all)
        meanF_local_all = np.mean(fluxes_local_all)
        meanF_eff_all = np.mean(fluxes_eff_all)

    # calculate effective optical depth wrt mean flux values
    tau_meanF_local_inbounds = -np.log(meanF_all_inbounds)
    tau_meanF_eff_inbounds = -np.log(meanF_eff_inbounds)
    if not outfile_exists:
        tau_meanF_local = -np.log(meanF_local_all)
        tau_meanF_eff = -np.log(meanF_eff_all)

    if args.verbose:
        curr_str = f'--- Mean local flux (inbounds) : {meanF_local_inbounds:.4e} / tau : {teau_meanF_local_inbounds:.4e} ---'
        curr_str = f'--- Mean effective flux (inbounds) : {meanF_eff_inbounds:.4e} / tau : {teau_meanF_eff_inbounds:.4e} ---'
        print(curr_str)

    # find the index of the skewers that do not fall within the input range & print its info
    indx_all_outbounds = np.argwhere(~tau_eff_all_inbounds_mask).flatten()
    nskews_outbounds = np.sum(~tau_eff_all_inbounds_mask)

    if args.verbose and nskews_outQuantiles:
        curr_str = f"--- We have {nskews_outquantiles:.0f} / {nskewers_tot:.0f} = "
        curr_str += f"{100 * nskews_outquantiles / nskewers_tot:.0f} % skewers outside of bounds ---"
        print(curr_str)

    if args.verbose:
        print(f"--- Calculating cosmology information for flux power spectrum ---")

    # calculate Hubble flow across a cell
    chCosmoHead = ChollaCosmologyHead(Omega_M, Omega_R, Omega_K, Omega_L, w0, wa, H0)
    chSnapCosmoHead = ChollaSnapCosmologyHead(scale_factor, chCosmoHead)
    dvHubble_x = chSnapCosmoHead.dvHubble(OTFSkewers.dx)
    dvHubble_y = chSnapCosmoHead.dvHubble(OTFSkewers.dy)
    dvHubble_z = chSnapCosmoHead.dvHubble(OTFSkewers.dz)

    # initialize flux power spectrum
    FPS_x = np.zeros(nFFTs[0], dtype=precision)
    FPS_y = np.zeros(nFFTs[1], dtype=precision)
    FPS_z = np.zeros(nFFTs[2], dtype=precision)

    # create Flux Power Spectrum object
    FPSHead_x = ChollaFluxPowerSpectrumHead(nCells[0], dvHubble_x)
    FPSHead_y = ChollaFluxPowerSpectrumHead(nCells[1], dvHubble_y)
    FPSHead_z = ChollaFluxPowerSpectrumHead(nCells[2], dvHubble_z)

    if not outfile_exists:
        # calculate kmodes
        kvals_fft_x = FPSHead_x.get_kvals_fft(dtype=precision)
        kvals_fft_y = FPSHead_y.get_kvals_fft(dtype=precision)
        kvals_fft_z = FPSHead_z.get_kvals_fft(dtype=precision)

    if args.verbose:
        print("--- Found k-modes along each axis, now moving on to actually performing flux power spectrum calculation in each quantile ---")


    if nskews_x_inbounds:
        _, calc_FPS_x = FPSHead_x.get_FPS(tau_local_x_inbounds,
                                          flux_mean_global=meanF_local_inbounds, 
                                          precision=precision)
        FPS_x += calc_FPS_x

    if nskews_y_inbounds:
        _, calc_FPS_y = FPSHead_y.get_FPS(tau_local_y_inbounds,
                                          flux_mean_global=meanF_local_inbounds,
                                          precision=precision)
        FPS_y += calc_FPS_y

    if nskews_z_inbounds:    
        _, calc_FPS_z = FPSHead_z.get_FPS(tau_local_z_inbounds,
                                          flux_mean_global=meanF_local_inbounds,
                                          precision=precision)
        FPS_z += calc_FPS_z

    
    if args.verbose:
        print(f"--- Flux power spectrum calculation completed along each axis ! Now saving data ---")

    with h5py.File(outfile_fPath, 'a') as fObj:
        # place attributes
        if not outfile_exists:
            # start with cosmo info
            _ = fObj.attrs.create('Omega_R', Omega_R)
            _ = fObj.attrs.create('Omega_M', Omega_M)
            _ = fObj.attrs.create('Omega_L', Omega_L)
            _ = fObj.attrs.create('Omega_K', Omega_K)
            _ = fObj.attrs.create('Omega_b', Omega_b)
            _ = fObj.attrs.create('w0', w0)
            _ = fObj.attrs.create('wa', wa)
            _ = fObj.attrs.create('H0', H0)

            # sim info
            _ = fObj.attrs.create('Lbox', Lbox)
            _ = fObj.attrs.create('nCells', nCells)
            _ = fObj.attrs.create('nStrides', nstrides)
            _ = fObj.attrs.create('nSkewers', np.array([nskewers_x, nskewers_y, nskewers_z]))

            # output info
            _ = fObj.attrs.create('fPath', str(skewer_fPath))
            _ = fObj.attrs.create('redshift', redshift)
            _ = fObj.attrs.create('scale_factor', scale_factor)
            _ = fObj.attrs.create('nOutput', nOutput)

            # analysis info
            _ = fObj.create_dataset('k_x', data=kvals_fft_x)
            _ = fObj.create_dataset('k_y', data=kvals_fft_y)
            _ = fObj.create_dataset('k_z', data=kvals_fft_z)
            _ = fObj.attrs.create('nranges', 0)
            _ = fObj.attrs.create('nquantiles', 0)
            _ = fObj.attrs.create('tau_local_mean', tau_local_all_mean)
            _ = fObj.attrs.create('tau_eff_mean', tau_eff_all_mean)
            _ = fObj.attrs.create('meanF_local', meanF_local_all)
            _ = fObj.attrs.create('meanF_eff', meanF_eff_all)
            _ = fObj.attrs.create('tau_meanF_local', tau_meanF_local)
            _ = fObj.attrs.create('tau_meanF_eff', tau_meanF_eff)

        curr_nranges = fObj.attrs['nranges'].item()
        range_key = f'range_{curr_nranges:.0f}'
        range_group = fObj.create_group(range_key)
        
        _ = range_group.attrs.create('tau_min', args.optdepthlow)
        _ = range_group.attrs.create('tau_max', args.optdepthupp)
        _ = range_group.attrs.create('tau_mean', np.mean(tau_eff_all[tau_eff_all_inbounds_mask]))

        _ = range_group.attrs.create('meanF_local_inrange', meanF_local_inbounds)
        _ = range_group.attrs.create('meanF_eff_inrange', meanF_eff_inbounds)

        _ = range_group.attrs.create('tau_meanF_local_inrange', tau_meanF_local_inbounds)
        _ = range_group.attrs.create('tau_meanF_eff_inrange', tau_meanF_eff_inbounds)

        _ = range_group.create_dataset('indices', data=indx_all_inbounds)
        _ = range_group.create_dataset('FPS_x', data=FPS_x)
        _ = range_group.create_dataset('FPS_y', data=FPS_y)
        _ = range_group.create_dataset('FPS_z', data=FPS_z)
        
        _ = fObj.attrs.modify('nranges', int(curr_nranges+1))



    if args.verbose:
        print(f"--- Done ! ---")


if __name__=="__main__":
    main()


