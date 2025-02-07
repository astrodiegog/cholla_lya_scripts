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
    Create a command line argument parser that grabs the number of nodes
        and the parameter text file. Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Compute and append power spectra")

    parser.add_argument("skewdirname", help='Cholla skewer output directory name', type=str)

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument("nquantiles", help='Number of quantiles to bin the optical depths', type=int)

    parser.add_argument("optdepthlow", help='Lower effective optical depth limit to bin', type=float)

    parser.add_argument("optdepthupp", help='Upper effective optical depth limit to bin', type=float)

    parser.add_argument("nOutputsStr", help='String of outputs delimited by comma', type=str)

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

    def get_FPS(self, local_opticaldepths, mean_flux=None, precision=np.float64):
        '''
        Return the Flux Power Spectrum given the local optical depths.
            Expect 2-D array of shape (number skewers, line-of-sight cells)

        Args:
            local_opticaldepths (arr): local optical depths of all skewers
            mean_flux (float): mean flux to scale deviations
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
        if mean_flux:
            flux_mean = mean_flux
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
        Check skewer group to set the available keys

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

    def get_alllocalopticaldepth(self, dtype=np.float32):
        '''
        Return local optical depth array for all skewers

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            arr (arr): local optical depth
        '''

        return self.get_skeweralldata(self.local_opticaldepth_key, dtype=dtype)




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

    def get_FPS_x(self, dlogk, precision=np.float64):
        '''
        Return the Flux Power Spectrum along the x-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab x-skewer object
        OTFSkewers_x = self.get_skewersx_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_i.get_alllocalopticaldepth(precision)

        # create power spectrum object with x-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.nx, self.dvHubble_x)

        # return flux power spectrum along x-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision)

    def get_FPS_y(self, dlogk, precision=np.float64):
        '''
        Return the Flux Power Spectrum along the y-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab y-skewer object
        OTFSkewers_y = self.get_skewersy_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_i.get_alllocalopticaldepth(precision)

        # create power spectrum object with y-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.ny, self.dvHubble_y)

        # return flux power spectrum along y-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision)

    def get_FPS_z(self, dlogk, precision=np.float64):
        '''
        Return the Flux Power Spectrum along the z-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab z-skewer object
        OTFSkewers_z = self.get_skewersz_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_i.get_alllocalopticaldepth(precision)

        # create power spectrum object with y-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.nz, self.dvHubble_z)

        # return flux power spectrum along z-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision)


def main():
    '''
    Compute the power spectrum and append to skewer file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer directory : {args.skewdirname} ---")
        print(f"--- We are placing power spectra in dlogk : {args.dlogk} ---")
        
        print(f"--- We are binning effective optical depth in {args.nquantiles:.0f} quantiles ---")
        print(f"--- We have a range from {args.optdepthlow:.3e} to {args.optdepthupp:.3e}---")

        print(f"--- We will be using outputs : {args.nOutputsStr} ---")
        

    precision = np.float64

    # make sure directories exist
    skewer_dirPath = Path(args.skewdirname).resolve()
    assert skewer_dirPath.is_dir()


    # parse nOutputsStr, get list of outputs, convert to np array
    nOutputsStr_lst = args.nOutputsStr.split(',')
    nOutputs_arr = np.zeros(len(nOutputsStr_lst), dtype=np.int64)
    for n, nOutputStr in enumerate(nOutputsStr_lst):
        # cast into int
        nOutputs_arr[n] = int(nOutputStr)
    nOutputs = nOutputs_arr.size

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    # ensure number of quantiles is reasonable
    assert args.nquantiles > 0

    # ensure limits are reasonable
    assert args.optdepthlow >= 0
    assert args.optdepthupp > args.optdepthlow

    # go over each file to ...
    # make sure required keys are there
    # get data to save as future attrs
    # get skewer specific information

    req_keys = ['taucalc_local', 'taucalc_eff']
    precision = np.float64
    Omega_K, Omega_L, Omega_M, Omega_R, Omega_b = 0., 0., 0., 0., 0.
    H0, w0, wa = 0., 0., 0.
    Lbox = np.zeros(3, dtype=precision)
    nCells = np.zeros(3, dtype=precision)
    scale_factors = np.zeros(nOutputs, dtype=precision)
    redshifts = np.zeros(nOutputs, dtype=precision)
    nstrides = np.zeros(3, dtype=precision)
    nskewers_x, nskewers_y, nskewer_z = 0, 0, 0

    for n, nOutput in enumerate(nOutputs_arr):
        skewer_fname = f"{nOutput:.0f}_skewers.h5"
        skewer_fPath = skewer_dirPath / Path(skewer_fname)
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

        # save snapshot specific information
        scale_factors[n] = OTFSkewers.current_a
        redshifts[n] = OTFSkewers.current_z

        #  assume that the following attributes are constant, save first go around
        if not n:
            Omega_K, Omega_L = OTFSkewers.Omega_K, OTFSkewers.Omega_L
            Omega_M, Omega_R = OTFSkewers.Omega_M, OTFSkewers.Omega_R
            Omega_b, H0 = OTFSkewers.Omega_b, OTFSkewers.H0
            w0, wa = OTFSkewers.w0, OTFSkewers.wa
            Lbox[0] = OTFSkewers.dx * OTFSkewers.nx
            Lbox[1] = OTFSkewers.dy * OTFSkewers.ny
            Lbox[2] = OTFSkewers.dz * OTFSkewers.nz
            nCells[0] = OTFSkewers.nx
            nCells[1] = OTFSkewers.ny
            nCells[2] = OTFSkewers.nz
            nstrides[0] = OTFSkewers.nstride_x
            nstrides[1] = OTFSkewers.nstride_y
            nstrides[2] = OTFSkewers.nstride_z

            nskewers_x = int((OTFSkewers.ny * OTFSkewers.nz) / (OTFSkewers.nstride_x * OTFSkewers.nstride_x))
            nskewers_y = int((OTFSkewers.nx * OTFSkewers.nz) / (OTFSkewers.nstride_y * OTFSkewers.nstride_y))
            nskewers_z = int((OTFSkewers.nx * OTFSkewers.ny) / (OTFSkewers.nstride_z * OTFSkewers.nstride_z))
    
    nskewers_tot = nskewers_x + nskewers_y + nskewers_z
    if args.verbose:
        print(f"--- We are sorting {nskewers_tot * nOutputs:.0f} total skewers ---")


    # make arrays of all effective optical depths along each axis
    # knowing the axis_skewid and nOutput, index will be [nOutput * nskewers_x + axis_skewid]   
    optdeptheff_x = np.zeros(nOutputs * nskewers_x, dtype=precision)
    optdeptheff_y = np.zeros(nOutputs * nskewers_y, dtype=precision)
    optdeptheff_z = np.zeros(nOutputs * nskewers_z, dtype=precision)
    for n, nOutput in enumerate(nOutputs_arr):
        skewer_fname = f"{nOutput:.0f}_skewers.h5"
        skewer_fPath = skewer_dirPath / Path(skewer_fname)
        # create ChollaOTFSkewers object
        OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)
        OTFSkewers_x = OTFSkewers.get_skewersx_obj()
        OTFSkewers_y = OTFSkewers.get_skewersy_obj()
        OTFSkewers_z = OTFSkewers.get_skewersz_obj()
        optdeptheff_x[n * nskewers_x: (n+1)*nskewers_x] = OTFSkewers_x.get_skeweralldata('taucalc_eff', 
                                                                                         dtype=precision)
        optdeptheff_y[n * nskewers_y: (n+1)*nskewers_y] = OTFSkewers_y.get_skeweralldata('taucalc_eff', 
                                                                                         dtype=precision)
        optdeptheff_z[n * nskewers_z: (n+1)*nskewers_z] = OTFSkewers_z.get_skeweralldata('taucalc_eff', 
                                                                                         dtype=precision)

    # group all effective optical depths
    optdeptheff_all = np.zeros(nOutputs * nskewers_tot, dtype=precision)
    optdeptheff_all[ : nOutputs * (nskewers_x)] = optdeptheff_x
    optdeptheff_all[nOutputs * (nskewers_x) : nOutputs * (nskewers_x + nskewers_y)] = optdeptheff_y
    optdeptheff_all[nOutputs * (nskewers_x + nskewers_y) : ] = optdeptheff_z

    # create a mask of all effective optical depths within our range
    optdeptheff_all_inbounds_mask = (optdeptheff_all > args.optdepthlow) & (optdeptheff_all < args.optdepthupp)
    nskewers_inbounds = np.sum(optdeptheff_all_inbounds_mask)

    # calculate the number of skewers that is going to fall within each quantile
    nskewers_perquantile = nskewers_inbounds / args.nquantiles

    # index of (0, nOutputs * nskewers_x) corresponds to x-axis
    # index of (nOutputs * nskewers_x, nOutputs * (nskewers_x + nskewers_y)) corresponds to y-axis
    # index of (nOutputs * (nskewers_x + nskewers_y), nOutputs *(nskewers_x + nskewers_y + nskewers_z)) corresponds to z-axis
    indices_all_optdepthsort = np.argsort(optdeptheff_all)
    indices_all_inbounds_mask_optdepthsort = optdeptheff_all_inbounds_mask[indices_all_optdepthsort]
  
    # array of indices that fall within quantile
    indices_all_optdepthsort_inbounds = indices_all_optdepthsort[indices_all_inbounds_mask_optdepthsort]

    # create index that places each sorted opt depth into a quantile
    indices_all_inbounds_arange_quantile = np.floor(np.arange(nskewers_inbounds) / nskewers_perquantile)

    # create index directories for each quantile. may not have same number of skewers in each quantile
    indices_all_quantiles = {}

    # calculate the mean flux in each quantile
    meanF_all_quantiles = {}
    for nquantile in range(args.nquantiles):
        indices_all_inbounds_inquantile = nquantile == indices_all_inbounds_arange_quantile
        quantile_key = f'quantile_{nquantile:.0f}'
        indices_all_quantiles[quantile_key] = indices_all_optdepthsort_inbounds[indices_all_inbounds_inquantile]
    
        quantile_indx = indices_all_optdepthsort_inbounds[indices_all_inbounds_inquantile]
        nskews_inquantile = np.sum(indices_all_inbounds_inquantile)
        
        # make masks of the indices that fall within a specific axis
        quantile_indx_x_mask = quantile_indx < nOutputs * (nskewers_x)
        quantile_indx_y_mask = ( nOutputs * (nskewers_x) < quantile_indx) & (quantile_indx < nOutputs * (nskewers_x + nskewers_y))
        quantile_indx_z_mask = ( nOutputs * (nskewers_x + nskewers_y) < quantile_indx)

        nskewsx_inquantile = np.sum(quantile_indx_x_mask)
        nskewsy_inquantile = np.sum(quantile_indx_y_mask)
        nskewsz_inquantile = np.sum(quantile_indx_z_mask)

        # apply masks to get indices in each axis
        indx_x_currQuantile = quantile_indx[quantile_indx_x_mask]
        indx_y_currQuantile = quantile_indx[quantile_indx_y_mask]
        indx_z_currQuantile = quantile_indx[quantile_indx_z_mask]

        # calculate the output number that each index occupies
        indx_x_currQuantile_nOutput = indx_x_currQuantile // nskewers_x
        indx_y_currQuantile_nOutput = (indx_y_currQuantile - nOutputs * (nskewers_x)) // nskewers_y
        indx_z_currQuantile_nOutput = (indx_z_currQuantile - nOutputs * (nskewers_x + nskewers_y))  // nskewers_z

        if args.verbose:
            print(f"--- Distribution of {nskews_inquantile:.0f} skewers in quantile {nquantile:.0f}  ---")
            print(f"--- tau = [{optdeptheff_all[quantile_indx[0]]:.4e}, {optdeptheff_all[quantile_indx[-1]]:.4e}] ---")
            print("--- | n | nOutput | scale factor | redshift | x_skewers | y_skewers | z_skewers | ---")

        # create an array for all local optical depths along an axis
        tau_local_x_inquantile = np.zeros(int(nskewsx_inquantile * nCells[0]))
        tau_local_y_inquantile = np.zeros(int(nskewsy_inquantile * nCells[1]))
        tau_local_z_inquantile = np.zeros(int(nskewsz_inquantile * nCells[2]))        

        nskewsx_counter = 0
        nskewsy_counter = 0
        nskewsz_counter = 0

        for n, nOutput in enumerate(nOutputs_arr):
            # grab only indices in output
            indx_x_currQuantile_currOutput_mask = indx_x_currQuantile_nOutput == n
            indx_y_currQuantile_currOutput_mask = indx_y_currQuantile_nOutput == n
            indx_z_currQuantile_currOutput_mask = indx_z_currQuantile_nOutput == n

            indx_x_currQuantile_currOutput = indx_x_currQuantile[indx_x_currQuantile_currOutput_mask]
            indx_y_currQuantile_currOutput = indx_y_currQuantile[indx_y_currQuantile_currOutput_mask]
            indx_z_currQuantile_currOutput = indx_z_currQuantile[indx_z_currQuantile_currOutput_mask]

            # calculate the number of skewers that land in this quantile in this output
            nskewsx_inOutput_inQuantile = np.sum(indx_x_currQuantile_currOutput_mask)
            nskewsy_inOutput_inQuantile = np.sum(indx_y_currQuantile_currOutput_mask)
            nskewsz_inOutput_inQuantile = np.sum(indx_z_currQuantile_currOutput_mask)

            if args.verbose:
                curr_str = f"--- | {n:.0f} | {nOutput:.0f} | {scale_factors[n]:.4f} | {redshifts[n]:.4f} | "
                curr_str += f"{100. * nskewsx_inOutput_inQuantile / nskews_inquantile:.4f} % | "
                curr_str += f"{100. * nskewsy_inOutput_inQuantile / nskews_inquantile:.4f} % | "
                curr_str += f"{100. * nskewsz_inOutput_inQuantile / nskews_inquantile:.4f} % | ---"

                print(curr_str)

            if not (nskewsx_inOutput_inQuantile or nskewsy_inOutput_inQuantile or nskewsz_inOutput_inQuantile):
                continue

            # convert the index into a skewer id to index into local optical depth array
            skewid_currQuantile_currOutput_x = indx_x_currQuantile_currOutput % nskewers_x
            skewid_currQuantile_currOutput_y = indx_y_currQuantile_currOutput % nskewers_y
            skewid_currQuantile_currOutput_z = indx_z_currQuantile_currOutput % nskewers_z

            # create ChollaOTFSkewers object to grab local optical depths
            skewer_fname = f"{nOutput:.0f}_skewers.h5"
            skewer_fPath = skewer_dirPath / Path(skewer_fname)
            OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)
            OTFSkewers_x = OTFSkewers.get_skewersx_obj()
            OTFSkewers_y = OTFSkewers.get_skewersy_obj()
            OTFSkewers_z = OTFSkewers.get_skewersz_obj()

            tau_local_x = OTFSkewers_x.get_skeweralldata('taucalc_local', dtype=precision)
            tau_local_y = OTFSkewers_y.get_skeweralldata('taucalc_local', dtype=precision)
            tau_local_z = OTFSkewers_z.get_skeweralldata('taucalc_local', dtype=precision)

            # grab current output local optical depth in quantile
            tau_local_x_currQuantile_currOutput = tau_local_x[skewid_currQuantile_currOutput_x, :]
            tau_local_y_currQuantile_currOutput = tau_local_y[skewid_currQuantile_currOutput_y, :]
            tau_local_z_currQuantile_currOutput = tau_local_z[skewid_currQuantile_currOutput_z, :]

            # place local taus from current output into quantile array
            tau_local_x_inquantile[ int(nskewsx_counter * nCells[0]) : int((nskewsx_counter + nskewsx_inOutput_inQuantile) * nCells[0]) ] = tau_local_x_currQuantile_currOutput.flatten()
            tau_local_y_inquantile[ int(nskewsy_counter * nCells[1]) : int((nskewsy_counter + nskewsy_inOutput_inQuantile) * nCells[1]) ] = tau_local_y_currQuantile_currOutput.flatten()
            tau_local_z_inquantile[ int(nskewsz_counter * nCells[2]) : int((nskewsz_counter + nskewsz_inOutput_inQuantile) * nCells[2]) ] = tau_local_z_currQuantile_currOutput.flatten()

            nskewsx_counter += nskewsx_inOutput_inQuantile
            nskewsy_counter += nskewsy_inOutput_inQuantile
            nskewsz_counter += nskewsz_inOutput_inQuantile       
 

        nCells_inquantile = (nskewsx_inquantile * nCells[0]) + (nskewsy_inquantile * nCells[1]) + (nskewsz_inquantile * nCells[2])
        tau_local_inquantile = np.zeros(int(nCells_inquantile))
        tau_local_inquantile[ : int(nskewsx_inquantile * nCells[0]) ] = tau_local_x_inquantile
        tau_local_inquantile[int(nskewsx_inquantile * nCells[0]) :int((nskewsx_inquantile * nCells[0]) + (nskewsy_inquantile * nCells[1])) ] = tau_local_y_inquantile
        tau_local_inquantile[int((nskewsx_inquantile * nCells[0]) + (nskewsy_inquantile * nCells[1]) ): ] = tau_local_z_inquantile
        
        fluxes_local = np.exp(- tau_local_inquantile)
        meanF_all_quantiles[quantile_key] = np.mean(fluxes_local)

        if args.verbose:
            print(f"--- Mean Flux : {meanF_all_quantiles[quantile_key]:.4e} ---")
            print("\n")

    indices_out_quantiles = np.argwhere(~optdeptheff_all_inbounds_mask)
    nskews_outquantiles = np.sum(~optdeptheff_all_inbounds_mask) 

    if args.verbose and nskews_outquantiles:
        curr_str = f"--- We have {nskews_outquantiles:.0f} / {nskewers_tot * nOutputs:.0f} = "
        curr_str += f"{100 * nskews_outquantiles / (nskewers_tot * nOutputs):.0f} % skewers outside of bounds ---"
        print(curr_str)

        # calculate the output number that each index occupies
        indx_x_out = indices_out_quantiles // nskewers_x
        indx_y_out = (indices_out_quantiles - nOutputs * (nskewers_x)) // nskewers_y
        indx_z_out = (indices_out_quantiles - nOutputs * (nskewers_x + nskewers_y))  // nskewers_z

        print("--- | n | nOutput | scale factor | redshift | x_skewers | y_skewers | z_skewers |---")
        # get indices of those outside the range and print info here
        for n, nOutput in enumerate(nOutputs_arr):
            indx_x_out_currOutput_mask = indx_x_out == n
            indx_y_out_currOutput_mask = indx_y_out == n
            indx_z_out_currOutput_mask = indx_z_out == n
            curr_str = f"--- | {n:.0f} | {nOutput:.0f} | {scale_factors[n]:.4f} | {redshifts[n]:.4f} | "
            curr_str += f"{100. * np.sum(indx_x_out_currOutput_mask) / nskews_outquantiles:.4f} % | "
            curr_str += f"{100. * np.sum(indx_y_out_currOutput_mask) / nskews_outquantiles:.4f} % | "
            curr_str += f"{100. * np.sum(indx_z_out_currOutput_mask) / nskews_outquantiles:.4f} % | ---"
            print(curr_str)
        print("\n")


    # calculate Hubble flow to find most inclusive k_min or u_max
    chCosmoHead = ChollaCosmologyHead(Omega_M, Omega_R, Omega_K, Omega_L, w0, wa, H0)
    dx, dy, dz = Lbox / nCells
    dvHubbles_x = np.zeros(nOutputs, dtype=precision)
    dvHubbles_y = np.zeros(nOutputs, dtype=precision)
    dvHubbles_z = np.zeros(nOutputs, dtype=precision)

    for n, scale_factor in enumerate(scale_factors):
        chSnapCosmoHead = ChollaSnapCosmologyHead(scale_factor, chCosmoHead)
        dvHubbles_x[n] = chSnapCosmoHead.dvHubble(dx)
        dvHubbles_y[n] = chSnapCosmoHead.dvHubble(dy)
        dvHubbles_z[n] = chSnapCosmoHead.dvHubble(dz)

    dvHubblex_min, dvHubblex_max = np.min(dvHubbles_x), np.max(dvHubbles_x)
    dvHubbley_min, dvHubbley_max = np.min(dvHubbles_y), np.max(dvHubbles_y)
    dvHubblez_min, dvHubblez_max = np.min(dvHubbles_z), np.max(dvHubbles_z)

    dvHubble_min = np.min([dvHubblex_min, dvHubbley_min, dvHubblez_min])
    dvHubble_max = np.max([dvHubblex_max, dvHubbley_max, dvHubblez_max])

    u_max = dvHubble_max * np.max(nCells)
    l_kmin = np.log10( (2. * np.pi) / u_max)
    l_kmax = np.log10( (2. * np.pi) / dvHubble_min)

    if args.verbose:
        print("dvHubble min:", dvHubble_min)
        print("dvHubble max:", dvHubble_max)
        print("l_kmin: ", l_kmin)
        print("l_kmax: ", l_kmax)
        print("u_max: ", u_max)

    # create k value edges for inclusive power spectrum
    n_bins = int((l_kmax - l_kmin) / args.dlogk)
    iter_arr = np.arange(n_bins + 1)
    kedges = np.zeros(n_bins+1, dtype=precision)
    kedges[:] = 10**(l_kmin + (args.dlogk * iter_arr))  

    # initialize flux power spectrum and create mean flux for each quantile
    FPS_all_quantiles = {}
    FPS_nOutputs_quantiles = {}
    for nquantile in range(args.nquantiles):
        quantile_key = f'quantile_{nquantile:.0f}'
        FPS_all_quantiles[quantile_key] = np.zeros(n_bins, dtype=precision)
        indices_all_inbounds_inquantile = nquantile == indices_all_inbounds_arange_quantile
        quantile_indx = indices_all_optdepthsort_inbounds[indices_all_inbounds_inquantile]
        FPS_nOutputs_quantiles[quantile_key] = {}

    fft_kvals_nOutput = []
    fft_binids_nOutput = []
    fft_nbins_nOutput = []

    # create flux power spectrum for each quantile
    for n, nOutput in enumerate(nOutputs_arr):
        # create Flux Power Spectrum object
        FPSHead_x = ChollaFluxPowerSpectrumHead(nCells[0], dvHubbles_x[n])
        FPSHead_y = ChollaFluxPowerSpectrumHead(nCells[1], dvHubbles_y[n])
        FPSHead_z = ChollaFluxPowerSpectrumHead(nCells[2], dvHubbles_z[n])

        # calculate kmodes
        kvals_fft_x = FPSHead_x.get_kvals_fft(dtype=precision)
        kvals_fft_y = FPSHead_y.get_kvals_fft(dtype=precision)
        kvals_fft_z = FPSHead_z.get_kvals_fft(dtype=precision)

        # for each axis, calculate where kfft lands on kedges
        # then find out _how many_ fft modes land in each kedge bin
        fft_binids_float_x = (np.log10(kvals_fft_x) - l_kmin ) / args.dlogk
        fft_binids_x = np.zeros(FPSHead_x.n_fft, dtype=np.int64)
        fft_binids_x[:] = np.floor(fft_binids_float_x)
        fft_nbins_kedges_x = np.zeros(n_bins, dtype=precision)
        for fft_bin_id in fft_binids_x[1:]:
            fft_nbins_kedges_x[fft_bin_id] += 1.
        
        fft_kvals_nOutput.append(kvals_fft_x)
        fft_binids_nOutput.append(fft_binids_x)
        fft_nbins_nOutput.append(fft_nbins_kedges_x)

        fft_binids_float_y = (np.log10(kvals_fft_y) - l_kmin ) / args.dlogk
        fft_binids_y = np.zeros(FPSHead_y.n_fft, dtype=np.int64)
        fft_binids_y[:] = np.floor(fft_binids_float_y)
        fft_nbins_kedges_y = np.zeros(n_bins, dtype=precision)
        for fft_bin_id in fft_binids_y[1:]:
            fft_nbins_kedges_y[fft_bin_id] += 1.

        fft_binids_float_z = (np.log10(kvals_fft_z) - l_kmin ) / args.dlogk
        fft_binids_z = np.zeros(FPSHead_z.n_fft, dtype=np.int64)
        fft_binids_z[:] = np.floor(fft_binids_float_z)
        fft_nbins_kedges_z = np.zeros(n_bins, dtype=precision)
        for fft_bin_id in fft_binids_z[1:]:
            fft_nbins_kedges_z[fft_bin_id] += 1.

        # (protect against dividing by zero)
        fft_nbins_kedges_x[fft_nbins_kedges_x == 0] = 1.
        fft_nbins_kedges_y[fft_nbins_kedges_y == 0] = 1.
        fft_nbins_kedges_z[fft_nbins_kedges_z == 0] = 1.

        # create ChollaOTFSkewers object to grab local optical depths
        skewer_fname = f"{nOutput:.0f}_skewers.h5"
        skewer_fPath = skewer_dirPath / Path(skewer_fname)
        OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)
        OTFSkewers_x = OTFSkewers.get_skewersx_obj()
        OTFSkewers_y = OTFSkewers.get_skewersy_obj()
        OTFSkewers_z = OTFSkewers.get_skewersz_obj()
        
        tau_local_x = OTFSkewers_x.get_skeweralldata('taucalc_local', dtype=precision)
        tau_local_y = OTFSkewers_y.get_skeweralldata('taucalc_local', dtype=precision)
        tau_local_z = OTFSkewers_z.get_skeweralldata('taucalc_local', dtype=precision)      

        if args.verbose:
            curr_str = f'--- Distribution of skewers in nOutput {nOutput} / scale factor: '
            curr_str += f'{scale_factors[n]:.4f} / redshift: {redshifts[n]:.4f} --- ' 
            print(curr_str)
            print(f'--- | nquantile | x_skewers | y_skewers | z_skewers | ---')

        for nquantile in range(args.nquantiles):
            indices_all_inbounds_inquantile = nquantile == indices_all_inbounds_arange_quantile
            quantile_key = f'quantile_{nquantile:.0f}'
            quantile_indx = indices_all_optdepthsort_inbounds[indices_all_inbounds_inquantile]
            # initialize
            FPS_nOutputs_quantiles[quantile_key][f'FPSx_nOutput_{nOutput:.0f}'] = np.zeros(FPSHead_x.n_fft)
            FPS_nOutputs_quantiles[quantile_key][f'FPSy_nOutput_{nOutput:.0f}'] = np.zeros(FPSHead_y.n_fft)
            FPS_nOutputs_quantiles[quantile_key][f'FPSz_nOutput_{nOutput:.0f}'] = np.zeros(FPSHead_z.n_fft)

            # make masks of the indices that fall within a specific axis
            quantile_indx_x_mask = quantile_indx < nOutputs * (nskewers_x)
            quantile_indx_y_mask = ( nOutputs * (nskewers_x) < quantile_indx) & (quantile_indx < nOutputs * (nskewers_x + nskewers_y))
            quantile_indx_z_mask = ( nOutputs * (nskewers_x + nskewers_y) < quantile_indx)

            # apply masks to get indices in each axis
            indx_x_currQuantile = quantile_indx[quantile_indx_x_mask]
            indx_y_currQuantile = quantile_indx[quantile_indx_y_mask]
            indx_z_currQuantile = quantile_indx[quantile_indx_z_mask]

            # calculate the output number that each index occupies
            indx_x_currQuantile_nOutput = indx_x_currQuantile // nskewers_x
            indx_y_currQuantile_nOutput = (indx_y_currQuantile - nOutputs * (nskewers_x)) // nskewers_y
            indx_z_currQuantile_nOutput = (indx_z_currQuantile - nOutputs * (nskewers_x + nskewers_y))  // nskewers_z

            # grab only indices in output
            indx_x_currQuantile_currOutput_mask = indx_x_currQuantile_nOutput == n
            indx_y_currQuantile_currOutput_mask = indx_y_currQuantile_nOutput == n
            indx_z_currQuantile_currOutput_mask = indx_z_currQuantile_nOutput == n

            indx_x_currQuantile_currOutput = indx_x_currQuantile[indx_x_currQuantile_currOutput_mask]
            indx_y_currQuantile_currOutput = indx_y_currQuantile[indx_y_currQuantile_currOutput_mask]
            indx_z_currQuantile_currOutput = indx_z_currQuantile[indx_z_currQuantile_currOutput_mask]

            if args.verbose:
                curr_str = f"--- | {nquantile:.0f} | "
                curr_str += f"{100 * np.sum(indx_x_currQuantile_currOutput_mask) / nskewers_tot:.4f} % | "
                curr_str += f"{100 * np.sum(indx_y_currQuantile_currOutput_mask) / nskewers_tot:.4f} % | "
                curr_str += f"{100 * np.sum(indx_z_currQuantile_currOutput_mask) / nskewers_tot:.4f} % | --- "
                print(curr_str)

            # convert the index into a skewer id to index into local optical depth array
            skewid_currQuantile_currOutput_x = indx_x_currQuantile_currOutput % nskewers_x 
            skewid_currQuantile_currOutput_y = indx_y_currQuantile_currOutput % nskewers_y
            skewid_currQuantile_currOutput_z = indx_z_currQuantile_currOutput % nskewers_z

            # grab current output local optical depth in quantile
            tau_local_x_currQuantile_currOutput = tau_local_x[skewid_currQuantile_currOutput_x, :]
            tau_local_y_currQuantile_currOutput = tau_local_y[skewid_currQuantile_currOutput_y, :]
            tau_local_z_currQuantile_currOutput = tau_local_z[skewid_currQuantile_currOutput_z, :]

            # initialize FPS array from this quantile to add
            FPS_currQuantile = np.zeros(n_bins, dtype=precision)

            # for each axis: 1) compute flux power spectrum in quantile, 2) use fft to kedge bin map
            # to add FPS, 3) average by num of fft bins in that kedge bin. IF there are local tau
            if tau_local_x_currQuantile_currOutput.size:
                _, FPS_currQuantile_x = FPSHead_x.get_FPS(tau_local_x_currQuantile_currOutput,
                                                          mean_flux=meanF_all_quantiles[quantile_key], 
                                                          precision=precision)
                FPS_currQuantile[fft_binids_x[1:]] += FPS_currQuantile_x[1:]
                FPS_currQuantile /= fft_nbins_kedges_x
                FPS_nOutputs_quantiles[quantile_key][f'FPSx_nOutput_{nOutput:.0f}'] += FPS_currQuantile_x

            if tau_local_y_currQuantile_currOutput.size:
                _, FPS_currQuantile_y = FPSHead_y.get_FPS(tau_local_y_currQuantile_currOutput,
                                                          mean_flux=meanF_all_quantiles[quantile_key],
                                                          precision=precision)
                FPS_currQuantile[fft_binids_y[1:]] += FPS_currQuantile_y[1:]
                FPS_currQuantile /= fft_nbins_kedges_y
                FPS_nOutputs_quantiles[quantile_key][f'FPSy_nOutput_{nOutput:.0f}'] += FPS_currQuantile_y

            if tau_local_z_currQuantile_currOutput.size:
                _, FPS_currQuantile_z = FPSHead_z.get_FPS(tau_local_z_currQuantile_currOutput,
                                                          mean_flux=meanF_all_quantiles[quantile_key],
                                                          precision=precision)
                FPS_currQuantile[fft_binids_z[1:]] += FPS_currQuantile_z[1:]
                FPS_currQuantile /= fft_nbins_kedges_z
                FPS_nOutputs_quantiles[quantile_key][f'FPSz_nOutput_{nOutput:.0f}'] += FPS_currQuantile_z

            # place FPS from current output's quantile
            FPS_all_quantiles[quantile_key] += FPS_currQuantile
        
        if args.verbose and nskews_outquantiles:
            # calculate the output number that each index occupies
            indx_x_out_currOutput_mask = indx_x_out == n
            indx_y_out_currOutput_mask = indx_y_out == n
            indx_z_out_currOutput_mask = indx_z_out == n
            curr_str = "--- | out | "
            curr_str += f"{100 * np.sum(indx_x_out_currOutput_mask) / nskewers_tot:.4f} % | "
            curr_str += f"{100 * np.sum(indx_y_out_currOutput_mask) / nskewers_tot:.4f} % | "
            curr_str += f"{100 * np.sum(indx_z_out_currOutput_mask) / nskewers_tot:.4f} % | "
            print(curr_str)

            print('\n')

    _ = '''
    for n, nOutput in enumerate(nOutputs_arr):
        fft_binids = fft_binids_nOutput[n]
        fft_kvals = fft_kvals_nOutput[n]
        print(f"--- nOutput {nOutput:.0f} ---")
        for n_fft in range(1, FPSHead_x.n_fft-1):
            print(f"fftid: {n_fft:.0f} \t binid: {fft_binids[n_fft]:.0f} \t kval: {fft_kvals[n_fft]:.4e} \t kedge: {kedges[fft_binids[n_fft]]:.4e} \t kedge+1: {kedges[fft_binids[n_fft + 1]]:.4e}")
        print('\n')

    curr_str = "--- | kedge | kedge+1 | "
    for n, nOutput in enumerate(nOutputs_arr):
        curr_str += f" nOutput - {nOutput:.0f} |"
    curr_str += "---"    
    print(curr_str)

    for nbin in range(n_bins):
        curr_str = f"--- | {kedges[nbin]:.4e} | {kedges[nbin+1]:.4e} | "
        for n, nOutput in enumerate(nOutputs_arr):
            fft_nbins = fft_nbins_nOutput[n]
            curr_str += f" {fft_nbins[nbin]:.0f} |"
        curr_str += " ---"
        print(curr_str)
    '''
 
    

    # write data to where skewer directory resides
    skewerParent_dirPath = skewer_dirPath.parent.resolve()
    outfile_fname = f"fluxpowerspectrum_optdepthbin.h5"
    outfile_fPath = skewerParent_dirPath / Path(outfile_fname)

    with h5py.File(outfile_fPath, 'w') as fObj:
        # place attributes
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
        _ = fObj.attrs.create('redshifts', redshifts)
        _ = fObj.attrs.create('scale_factors', scale_factors)
        _ = fObj.attrs.create('nOutputs', nOutputs_arr)

        # analysis info
        _ = fObj.attrs.create('dlogk', args.dlogk)
        _ = fObj.attrs.create('tau_eff_low', args.optdepthlow)
        _ = fObj.attrs.create('tau_eff_upp', args.optdepthupp)
        _ = fObj.attrs.create('nquantiles', args.nquantiles)
        _ = fObj.attrs.create('k_edges', kedges)

        # flux power spectra info
        for nquantile in range(args.nquantiles):
            indices_all_inbounds_inquantile = nquantile == indices_all_inbounds_arange_quantile
            quantile_key = f'quantile_{nquantile:.0f}'
            quantile_indx = indices_all_optdepthsort_inbounds[indices_all_inbounds_inquantile]
            FPS_currQuantile = FPS_all_quantiles[quantile_key]
            FPS_nOutputs_currQuantile = FPS_nOutputs_quantiles[quantile_key]

            # grab upper and lower effective optical depths
            optdeptheff_currQuantile_min = optdeptheff_all[quantile_indx[0]]
            optdeptheff_currQuantile_max = optdeptheff_all[quantile_indx[-1]]
            optdeptheff_currQuantile_mean = np.mean(optdeptheff_all[quantile_indx])

            quantile_groupkey = "FluxPowerSpectrum_" + quantile_key
            quantile_group = fObj.create_group(quantile_groupkey)

            _ = quantile_group.attrs.create('tau_min', optdeptheff_currQuantile_min)
            _ = quantile_group.attrs.create('tau_max', optdeptheff_currQuantile_max)
            _ = quantile_group.attrs.create('tau_mean', optdeptheff_currQuantile_mean)

            _ = quantile_group.create_dataset('P(k)', data=FPS_currQuantile)
            _ = quantile_group.create_dataset('indices', data=quantile_indx)

            for n, nOutput in enumerate(nOutputs_arr):
                FPSx_nOutput = FPS_nOutputs_quantiles[quantile_key][f'FPSx_nOutput_{nOutput:.0f}']
                FPSy_nOutput = FPS_nOutputs_quantiles[quantile_key][f'FPSy_nOutput_{nOutput:.0f}']
                FPSz_nOutput = FPS_nOutputs_quantiles[quantile_key][f'FPSz_nOutput_{nOutput:.0f}']
                _ = quantile_group.create_dataset(f'FPSx_nOutput_{nOutput:.0f}', data=FPSx_nOutput)
                _ = quantile_group.create_dataset(f'FPSy_nOutput_{nOutput:.0f}', data=FPSy_nOutput)
                _ = quantile_group.create_dataset(f'FPSz_nOutput_{nOutput:.0f}', data=FPSz_nOutput)


#placebo: check Lambda-CDM. split into different tau. normalization of the power spectra knows about tau, soi

#1. science movtivation
#2. what I did
#3. results
 


if __name__=="__main__":
    main()


