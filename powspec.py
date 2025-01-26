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

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument('-c', '--combine', help='Whether to combine power spectrum from each axis',
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


###
# Create all data structures to fully explain power spectrum calculation
# These data structures are pretty thorough, and not every line is readily needed
# but I prioritize readability over less lines of code
###


###
# Calculations related to the geometry along an axis for a power spectrum calculation
###
# ChollaFluxPowerSpectrumHead    --> hold nfft and methods to get related k-mode arrays

class ChollaFluxPowerSpectrumHead:
    '''
    Cholla Flux Power Spectrum Head
    
    Holds information regarding the power spectrum calculation

        Initialized with:
        - dlogk (float): differential step in log k-space
        - nlos (int): number of line-of-sight cells
        - dvHubble (float): differential Hubble flow velocity across a cell

    Values are returned in code units unless otherwise specified.
    '''
    def __init__(self, dlogk, nlos, dvHubble):
        self.dlogk = dlogk
        self.n_los = nlos
        self.n_fft = int(self.n_los / 2 + 1)
        self.dvHubble = dvHubble

        # calculate Hubble flow across entire box (max velocity)
        self.u_max = self.dvHubble * self.n_los

        self.l_kmin = np.log10( (2. * np.pi) / (self.u_max) )
        self.l_kmax = np.log10( (2. * np.pi * (self.n_fft - 1.) ) / (self.u_max) )
        self.l_kstart = np.log10(0.99) + self.l_kmin
        self.n_bins = int(1 + ( (self.l_kmax - self.l_kstart) / self.dlogk ) )


    def get_kvals(self, dtype=np.float32):
        '''
        Return the k-centers of the power spectrum

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            kcenters (arr): k mode centers array
        '''

        kcenters = np.zeros(self.n_bins, dtype=dtype)
        iter_arr = np.arange(self.n_bins, dtype=dtype)

        kcenters[:] = 10**(self.l_kstart + (self.dlogk) * (iter_arr + 0.5) )

        return kcenters

    def get_kvals_edges(self, dtype=np.float32):
        '''
        Return the k-edges of the power spectrum

        Args:
            dtype (np type): (optional) numpy precision to use
        Returns:
            kedges (arr): k mode edges array
        '''

        kedges = np.zeros(self.n_bins + 1, dtype=dtype)
        iter_arr = np.arange(self.n_bins + 1, dtype=dtype)

        kedges[:] = 10**(self.l_kstart + (self.dlogk * iter_arr) )

        return kedges

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

    def get_fft_binids(self, dtype_bin=np.int64, dtype_calc=np.float32, useforloop=True):
        '''
        Return the indices that the k-mode fft bins land on within kvals_edges

        Args:
            dtype_bin (np type): (optional) numpy precision to use for returned array
            dtype_calc (np type): (optional) numpy precision to use for calculations
            useforloop (bool): (optional) whether to use for-loop or not  
        Returns:
            fft_binids (arr): indices where fft k-mode lands wrt kvals edges
        '''

        fft_binids = np.zeros(self.n_fft, dtype=dtype_bin)

        # grab fft kvalues
        kvals_fft = self.get_kvals_fft(dtype=dtype_calc)

        if useforloop:
            # grab edges for comparison
            kvals_edges = self.get_kvals_edges(dtype=dtype_calc)
            for bin_id_fft in range(self.n_fft):
                edge_greater_fft = np.argwhere(kvals_fft[bin_id_fft] < kvals_edges).flatten()
                # okay to flatten bc we know kvals_fft and kvals_edges are 1D arrays
                if edge_greater_fft.size > 0 :
                    # ensure we're indexing into a non-empty array
                    fft_binids[bin_id_fft] = edge_greater_fft[0] - 1
        else:
            fft_binids_float = (np.log10(kvals_fft) - self.l_kstart) / self.dlogk
            fft_binids[:] = np.floor(fft_binids_float)

        return fft_binids

    def get_FPS(self, local_opticaldepths, precision=np.float64, updated_deltaFcalc=False):
        '''
        Return the Flux Power Spectrum given the local optical depths.
            Expect 2-D array of shape (number skewers, line-of-sight cells)

        Args:
            local_opticaldepths (arr): local optical depths of all skewers
            precision (np type): (optional) numpy precision to use
            updated_deltaFcalc (bool): (optional) whether to calculate delta F 
                by subtracting mean
        Return:
            kmode_edges (arr): k mode edges array
            P_k_mean (arr): mean transmitted flux power spectrum within kmode edges
        '''
        assert local_opticaldepths.ndim == 2
        assert local_opticaldepths.shape[1] == self.n_los

        n_skews = local_opticaldepths.shape[0]

        # find the indices that describe where the k-mode FFT bins fall within kval_edges (from dlogk)
        fft_binids = self.get_fft_binids(dtype_bin=np.int64, dtype_calc=np.float64)

        # find number of fft modes that fall in requested dlogk bin id (used later to average total power in dlogk bin)
        hist_n = np.zeros(self.n_bins, dtype=precision)
        for bin_id in fft_binids[1:]:
            hist_n[bin_id] += 1.
        # (protect against dividing by zero)
        hist_n[hist_n == 0] = 1.

        # calculate local transmitted flux (& its mean)
        fluxes = np.exp(-local_opticaldepths)
        flux_mean = np.mean(fluxes)

        # initialize total power array & temporary FFT array
        hist_PS_vals = np.zeros(self.n_bins, dtype=precision)
        P_k_tot = np.zeros(self.n_bins, dtype=precision)

        for nSkewerID in range(n_skews):
            # clean out temporary FFT array
            hist_PS_vals[:] = 0.

            # calculate flux fluctuation 
            if updated_deltaFcalc:
                dFlux_skew = (fluxes[nSkewerID] - flux_mean) / flux_mean
            else:
                dFlux_skew = fluxes[nSkewerID] / flux_mean

            # perform fft & calculate amplitude of fft
            fft = np.fft.rfft(dFlux_skew)
            fft2 = (fft.imag * fft.imag + fft.real * fft.real) / self.n_los / self.n_los

            # add power for each fft mode
            hist_PS_vals[fft_binids[1:]] += fft2[1:]

            # take avg & scale by umax
            delta_F_avg = hist_PS_vals / hist_n
            P_k = self.u_max * delta_F_avg
            P_k_tot += P_k

        # average out by the number of skewers
        P_k_mean = P_k_tot / n_skews

        # grab k-mode bin edges
        kmode_edges = self.get_kvals_edges(precision)

        return (kmode_edges, P_k_mean)


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
        self.local_opticaldepth_key = 'taucalc_local'

        self.allkeys = {self.HI_str, self.HeII_str, self.density_str,
                        self.vel_str, self.temp_str, self.local_opticaldepth_key}

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

    def get_FPS_x(self, dlogk, precision=np.float64, updated_deltaFcalc=False):
        '''
        Return the Flux Power Spectrum along the x-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
            updated_deltaFcalc (bool): (optional) whether to calculate delta F 
                by subtracting mean
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab x-skewer object
        OTFSkewers_x = self.get_skewersx_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_x.get_alllocalopticaldepth(precision)

        # create power spectrum object with x-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.nx, self.dvHubble_x)

        # return flux power spectrum along x-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision, updated_deltaFcalc)

    def get_FPS_y(self, dlogk, precision=np.float64, updated_deltaFcalc=False):
        '''
        Return the Flux Power Spectrum along the y-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
            updated_deltaFcalc (bool): (optional) whether to calculate delta F 
                by subtracting mean
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab y-skewer object
        OTFSkewers_y = self.get_skewersx_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_y.get_alllocalopticaldepth(precision)

        # create power spectrum object with x-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.ny, self.dvHubble_y)

        # return flux power spectrum along x-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision, updated_deltaFcalc)

    def get_FPS_z(self, dlogk, precision=np.float64, updated_deltaFcalc=False):
        '''
        Return the Flux Power Spectrum along the z-axis

        Args:
            dlogk (float): differential step in log k-space
            precision (np type): (optional) numpy precision to use
            updated_deltaFcalc (bool): (optional) whether to calculate delta F 
                by subtracting mean
        Return:
            (arr): k mode edges array
            (arr): mean transmitted flux power spectrum within kmode edges
        '''
        # grab z-skewer object
        OTFSkewers_z = self.get_skewersx_obj()

        # grab local optical depths
        local_opticaldepth = OTFSkewers_z.get_alllocalopticaldepth(precision)

        # create power spectrum object with x-axis geometry
        FluxPowerSpectrumHead = ChollaFluxPowerSpectrumHead(dlogk, self.nz, self.dvHubble_z)

        # return flux power spectrum along x-axis
        return FluxPowerSpectrumHead.get_FPS(local_opticaldepth, precision, updated_deltaFcalc)



def P_k_calc(OTFSkewers, dlogk, combine=True, verbose=False, precision=np.float64, updated_deltaFcalc=False):
    '''
    Calculate the mean transmitted flux power spectrum along each axis and save
        onto skewer output file

    Args:
        OTFSkewers (ChollaOnTheFlySkewers): skewers object, interacts with files
        dlogk (float): differential step in log k-space
        combine (bool): (optional) whether to combine power spectrum from each axis
        verbose (bool): (optional) whether to print important information
        precision (np type): (optional) numpy precision to use in calculations
        updated_deltaFcalc (bool): (optional) whether to calculate delta F 
                by subtracting mean
    Returns:
        ...
    '''

    k_x, P_k_x = OTFSkewers.get_FPS_x(dlogk, precision, updated_deltaFcalc)
    k_y, P_k_y = OTFSkewers.get_FPS_y(dlogk, precision, updated_deltaFcalc)
    k_z, P_k_z = OTFSkewers.get_FPS_z(dlogk, precision, updated_deltaFcalc)

    # open file and append each power spectrum as new "PowerSpectrum" group
    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+') as fObj:
        if updated_deltaFcalc:
            PS_group_key = 'PowerSpectrum_newDeltaFCalc'
        else:
            PS_group_key = 'PowerSpectrum'
        if PS_group_key not in fObj.keys():
            if verbose:
                print(f'\t...initializing power spectrum group for file {OTFSkewers.OTFSkewersfPath}')

            fObj.create_group(PS_group_key)

        PS_x_avg_key = 'P_x(k_x)'
        k_x_edges_key = 'k_x_edges'
        PS_y_avg_key = 'P_y(k_y)'
        k_y_edges_key = 'k_y_edges'
        PS_z_avg_key = 'P_z(k_z)'
        k_z_edges_key = 'k_z_edges'

        # ensure each key is in the PowerSpectrum Group
        if PS_x_avg_key not in fObj[PS_group_key].keys():
            if verbose:
                print(f'\t...initializing empty power spectrum and k mode arrays in x-axis for file {OTFSkewers.OTFSkewersfPath}')

            PS_x_empty = np.zeros(P_k_x.shape, dtype=P_k_x.dtype)
            k_x_empty = np.zeros(k_x.shape, dtype=k_x.dtype)
            fObj[PS_group_key].create_dataset(k_x_edges_key, data=k_x_empty)
            fObj[PS_group_key].create_dataset(PS_x_avg_key, data=PS_x_empty)

        if PS_y_avg_key not in fObj[PS_group_key].keys():
            if verbose:
                print(f'\t...initializing empty power spectrum and k mode arrays in y-axis for file {OTFSkewers.OTFSkewersfPath}')

            PS_y_empty = np.zeros(P_k_y.shape, dtype=P_k_y.dtype)
            k_y_empty = np.zeros(k_y.shape, dtype=k_y.dtype)
            fObj[PS_group_key].create_dataset(k_y_edges_key, data=k_y_empty)
            fObj[PS_group_key].create_dataset(PS_y_avg_key, data=PS_y_empty)

        if PS_z_avg_key not in fObj[PS_group_key].keys():
            if verbose:
                print(f'\t...initializing empty power spectrum and k mode arrays in z-axis for file {OTFSkewers.OTFSkewersfPath}')
            PS_z_empty = np.zeros(P_k_z.shape, dtype=P_k_z.dtype)
            k_z_empty = np.zeros(k_z.shape, dtype=k_z.dtype)
            fObj[PS_group_key].create_dataset(k_z_edges_key, data=k_z_empty)
            fObj[PS_group_key].create_dataset(PS_z_avg_key, data=PS_z_empty)

        if verbose:
            print(f'\t...assigning power spectrum and k edges in each axis file {OTFSkewers.OTFSkewersfPath}')
        fObj[PS_group_key][k_x_edges_key][:] = k_x[:]
        fObj[PS_group_key][PS_x_avg_key][:] = P_k_x[:]

        fObj[PS_group_key][k_y_edges_key][:] = k_y[:]
        fObj[PS_group_key][PS_y_avg_key][:] = P_k_y[:]

        fObj[PS_group_key][k_z_edges_key][:] = k_z[:]
        fObj[PS_group_key][PS_z_avg_key][:] = P_k_z[:]

        if combine:
            # before combining power spectrum along each direction, make sure they're of same shape
            assert np.array_equal(P_k_x.shape, P_k_y.shape)
            assert np.array_equal(P_k_x.shape, P_k_z.shape)

            # also need to assert that k_xyz are all within some tolerance level
            # for now assume they're the same, and save k_x by default !

            # combined power spectrum
            P_k = (P_k_x + P_k_y + P_k_z) / 3.
            PS_avg_key = 'P(k)'
            k_edges_key = 'k_edges'
            if PS_avg_key not in fObj[PS_group_key].keys():
                if verbose:
                    print(f'\t...initializing empty power spectrum average and k mode arrays for file {OTFSkewers.OTFSkewersfPath}')
                PS_empty = np.zeros(P_k.shape, dtype=P_k.dtype)
                k_empty = np.zeros(k_x.shape, dtype=k_x.dtype)
                fObj[PS_group_key].create_dataset(k_edges_key, data=k_empty)
                fObj[PS_group_key].create_dataset(PS_avg_key, data=PS_empty)

            if verbose:
                print(f'\t...assigning average power spectrum and k edges for file {OTFSkewers.OTFSkewersfPath}')
            fObj[PS_group_key][k_edges_key][:] = k_x[:]
            fObj[PS_group_key][PS_avg_key][:] = P_k[:]

    return



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
        print(f"--- We are looking at skewer file : {args.skewfname} ---")
        print(f"--- We are placing power spectra in dlogk : {args.dlogk} ---")
        
        if args.combine:
            print(f"--- We are saving combined power spectra (!) ---")
        else:
            print(f"--- We are NOT saving combined power spectra (!) ---")

    precision = np.float64

    skewer_fPath = Path(args.skewfname).resolve()

    # ensure that local optical depth is a dataset
    with h5py.File(skewer_fPath, 'r') as fObj:    
        local_opticaldepth_key = 'taucalc_local'
        assert local_opticaldepth_key in fObj['skewers_x'].keys()
        assert local_opticaldepth_key in fObj['skewers_y'].keys()
        assert local_opticaldepth_key in fObj['skewers_z'].keys()

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)

    # calculate the power spectra with both methods
    P_k_calc(OTFSkewers, args.dlogk, args.combine, args.verbose, precision=np.float64)
    P_k_calc(OTFSkewers, args.dlogk, args.combine, args.verbose, precision=np.float64, updated_deltaFcalc=True)



if __name__=="__main__":
    main()


