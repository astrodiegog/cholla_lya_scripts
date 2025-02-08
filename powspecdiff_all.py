import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib.pyplot as plt


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
        description="Visualize power spectra usings different optical depth methods")

    parser.add_argument("skewdirname", help='Cholla skewer output directory name', type=str)

    parser.add_argument("analysisdirname", help='Cholla analysis output directory name', type=str)

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument("nOutput1", help='Output number for first plot in first row', type=int)
    parser.add_argument("nOutput2", help='Output number for second plot in first row', type=int)
    parser.add_argument("nOutput3", help='Output number for third plot in first row', type=int)
    
    parser.add_argument("nOutput4", help='Output number for first plot in second row', type=int)
    parser.add_argument("nOutput5", help='Output number for second plot in second row', type=int)
    parser.add_argument("nOutput6", help='Output number for third plot in second row', type=int)
    
    parser.add_argument("nOutput7", help='Output number for first plot in third row', type=int)
    parser.add_argument("nOutput8", help='Output number for second plot in third row', type=int)
    parser.add_argument("nOutput9", help='Output number for third plot in third row', type=int)

    parser.add_argument("nOutput10", help='Output number for first plot in fourth row', type=int)
    parser.add_argument("nOutput11", help='Output number for second plot in fourth row', type=int)
    parser.add_argument("nOutput12", help='Output number for third plot in fourth row', type=int)

    parser.add_argument('-l', '--logspace', help='Display relative difference in log-space', 
                        action='store_true')
    
    parser.add_argument('-f', '--fname', help='Output file name', type=str)

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser


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


    def get_FPS(self, local_opticaldepths, precision=np.float64):
        '''
        Return the Flux Power Spectrum given the local optical depths.
            Expect 2-D array of shape (number skewers, line-of-sight cells)

        Args:
            local_opticaldepths (arr): local optical depths of all skewers
            precision (np type): (optional) numpy precision to use
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
            dFlux_skew = fluxes[nSkewerID] / flux_mean

            # perform fft & calculate amplitude of fft
            fft = np.fft.rfft(dFlux_skew)
            fft2 = (fft.imag * fft.imag + fft.real * fft.real) / self.n_los / self.n_los

            # add power for each fft mode
            #hist_PS_vals[fft_binids[1:]] += fft2[1:]
            _ = np.add.at(hist_PS_vals, fft_binids[1:], fft2[1:])
    
            # take avg & scale by umax
            delta_F_avg = hist_PS_vals / hist_n
            P_k = self.u_max * delta_F_avg
            P_k_tot += P_k

        # average out by the number of skewers
        P_k_mean = P_k_tot / n_skews

        # grab k-mode bin edges
        kmode_edges = self.get_kvals_edges(precision)

        return (kmode_edges, P_k_mean)

###
# Skewer-specific information that interacts with skewers for a given skewer file
###
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

        # save cell distance in each direction to later calculate hubble flow
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


###
# Study specific functions
###

def plotFluxPowerSpectra_RelDiff(ax, k_model, Pk_model, Pk_tests, label_tests, log=False):
    '''
    Plot the relative difference of a set of flux power spectra
    
    Args:
        ax (Axes): matplotlib axis to place plot
        k_model (arr): k-centers of the flux power spectrum
        Pk_model (arr): flux power spectrum to compare against
        Pk_tests (list of arrs): arrays of testing flux power spectra
        label_tests (list of strs): strs to label each flux power spectra
        log (bool): whether to display difference in log-space
    Returns:
        ...
    '''
    # make sure we have one label for each FPS
    assert len(Pk_tests) == len(label_tests)
    # make sure all power spectra are of the same size
    for Pk_test in Pk_tests:
        assert k_model.size == Pk_test.size
    assert k_model.size == Pk_model.size

    # calculate dimensionless model flux power spectra to compare against
    delta2F_model = (1. / np.pi) * k_model * Pk_model

    for i, Pk_test in enumerate(Pk_tests):
        label = label_tests[i]
        delta2F_test = (1. / np.pi) * k_model * Pk_test
        delta2F_fracdiff = (delta2F_test - delta2F_model) / delta2F_model

        # no nans here !
        goodDelta2F = ~np.isnan(delta2F_fracdiff)
        if log:
            _ = ax.plot(k_model[goodDelta2F], np.abs(delta2F_fracdiff[goodDelta2F]), label=label)
        else:
            _ = ax.plot(k_model[goodDelta2F], delta2F_fracdiff[goodDelta2F], label=label)

    # add y limits
    if log:
        ylow, yupp = 1e-6, 1e-1
        _ = ax.set_yscale('log')
    else:
        ylow, yupp = -0.1, 0.1
    _ = ax.set_ylim(ylow, yupp)

    # add x lims
    xlow, xupp = 1e-3, 5e-2
    _ = ax.set_xlim(xlow, xupp)

    # set x log-scale
    _ = ax.set_xscale('log')

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    _ = ax.legend(fontsize=14, loc='upper left')

    return


def P_k_calc(OTFSkewers, dlogk, opticaldepth_key, verbose=False, precision=np.float64):
    '''
    Calculate the mean transmitted flux power spectrum along each axis and return the
        averaged power spectrum

    Args:
        OTFSkewers (ChollaOnTheFlySkewers): skewers object, interacts with files
        dlogk (float): differential step in log k-space
        opticaldepth_key (str): key to access optical depth within skewer group
        verbose (bool): (optional) whether to print important information
        precision (np type): (optional) numpy precision to use in calculations
    Returns:
        ...
    '''
    # Create Flux Power Spectrum header objects
    FPSHead_x = ChollaFluxPowerSpectrumHead(dlogk, OTFSkewers.nx, OTFSkewers.dvHubble_x)
    FPSHead_y = ChollaFluxPowerSpectrumHead(dlogk, OTFSkewers.ny, OTFSkewers.dvHubble_y)
    FPSHead_z = ChollaFluxPowerSpectrumHead(dlogk, OTFSkewers.nz, OTFSkewers.dvHubble_z)

    # allocate memory for mean P(k) along each axis
    Pk_x_mean = np.zeros(FPSHead_x.n_bins, dtype=precision)
    Pk_y_mean = np.zeros(FPSHead_y.n_bins, dtype=precision)
    Pk_z_mean = np.zeros(FPSHead_z.n_bins, dtype=precision)

    OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(), OTFSkewers.get_skewersy_obj(),
                      OTFSkewers.get_skewersz_obj()]

    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r') as fObj:
        for k, OTFSkewers_i in enumerate(OTFSkewers_lst):
            if (k==0):
                FPSHead_i = FPSHead_x
                Pk_i_mean = Pk_x_mean
            elif (k==1):
                FPSHead_i = FPSHead_y
                Pk_i_mean = Pk_y_mean
            elif (k==2):
                FPSHead_i = FPSHead_z
                Pk_i_mean = Pk_z_mean
            taus = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get(opticaldepth_key)[:]
            _, Pk_i_mean[:] = FPSHead_i.get_FPS(taus, precision=precision)

    Pk_avg = (Pk_x_mean + Pk_y_mean + Pk_z_mean) / 3.
    k_skew = FPSHead_i.get_kvals(dtype=precision)

    return k_skew, Pk_avg


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
        print(f"--- We are looking at skewer directory : {args.skewdirname} ---")
        print(f"--- We are looking at analysis directory : {args.analysisdirname} ---")
        print(f"--- Figure will look as ---")
        print(8*"---")
        print(f"---| {args.nOutput1:.0f} | {args.nOutput2:.0f} | {args.nOutput3:.0f} |---")
        print(f"---| {args.nOutput4:.0f} | {args.nOutput5:.0f} | {args.nOutput6:.0f} |---")
        print(f"---| {args.nOutput7:.0f} | {args.nOutput8:.0f} | {args.nOutput9:.0f} |---")
        print(f"---| {args.nOutput10:.0f} | {args.nOutput11:.0f} | {args.nOutput12:.0f} |---")
        print(8*"---")

    precision = np.float64

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    # make sure directories exist
    skewer_dirPath = Path(args.skewdirname).resolve()
    analysis_dirPath = Path(args.analysisdirname).resolve()
    assert skewer_dirPath.is_dir()
    assert analysis_dirPath.is_dir()


    all12_nOutput = np.zeros(12, dtype=np.int64)
    all12_nOutput[0], all12_nOutput[1], all12_nOutput[2] = args.nOutput1, args.nOutput2, args.nOutput3
    all12_nOutput[3], all12_nOutput[4], all12_nOutput[5] = args.nOutput4, args.nOutput5, args.nOutput6
    all12_nOutput[6], all12_nOutput[7], all12_nOutput[8] = args.nOutput7, args.nOutput8, args.nOutput9
    all12_nOutput[9], all12_nOutput[10], all12_nOutput[11] = args.nOutput10, args.nOutput11, args.nOutput12
    all12_nOutput = np.reshape(all12_nOutput, (4,3))

    # define optical depth keys to test against
    opticaldepth_testkeys = ['taucalc_local_3sig', 'taucalc_local_5sig', 'taucalc_local_8sig',
                             'taucalc_local_10sig', 'taucalc_local_12sig', 'taucalc_local_allLOS']

    fig, ax_all = plt.subplots(nrows=4, ncols=3, figsize=(12,15))
    ax_flat = ax_all.flatten()

    for i, row_nOutputs in enumerate(all12_nOutput):
        for j, nOutput in enumerate(row_nOutputs):
            analysisfname = f"{nOutput:.0f}_analysis.h5"
            skewersfname = f"{nOutput:.0f}_skewers.h5"            

            analysis_fPath = analysis_dirPath / Path(analysisfname)
            skewer_fPath = skewer_dirPath / Path(skewersfname)
            assert analysis_fPath.is_file()
            assert skewer_fPath.is_file()

            OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)
            ax = ax_all[i][j]    

            # get power spectra we're using to compare against
            Pk_analysis = np.array([])
            with h5py.File(analysis_fPath, 'r') as fObj_analysis:
                Pk_analysis = fObj_analysis['lya_statistics']['power_spectrum'].get('p(k)')[:]


            Pk_avg_tests = []
            testlabels = []
            for k, opticaldepth_key in enumerate(opticaldepth_testkeys):
                k_skew, Pk_avg = P_k_calc(OTFSkewers, args.dlogk, opticaldepth_key, args.verbose, precision)
                Pk_avg_tests.append(Pk_avg)

                if (k == 0):
                    numsigma = 3
                elif (k == 1):
                    numsigma = 5
                elif (k == 2):
                    numsigma = 8
                elif (k == 3):
                    numsigma = 10
                elif (k == 4):
                    numsigma = 12

                if (k < 5):
                    labelstr = rf'${numsigma:.0f} - b$'
                    testlabels.append(labelstr)
                else:
                    testlabels.append(r'$\rm{Entire}$ $\rm{LOS}$')

            plotFluxPowerSpectra_RelDiff(ax, k_skew, Pk_analysis, Pk_avg_tests, testlabels, args.logspace)

            # place labels
            if (i == 3):
                xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
            else:
                xlabel_str = None
                _ = ax.tick_params(labelbottom=False)
            if (j == 0):
                if args.logspace:
                    ylabel_str = r'$| D [ \Delta_F^2 (k) ] |$'
                else:
                    ylabel_str = r'$D [ \Delta_F^2 (k) ] $'
            else:
                ylabel_str = None
                _ = ax.tick_params(labelleft=False)

            _ = ax.set_xlabel(xlabel_str)
            _ = ax.set_ylabel(ylabel_str)

            # add redshift str
            xlow, xupp = 1e-3, 5e-2
            redshift_str = rf"$z = {OTFSkewers.current_z:.4f}$"
            x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
            if args.logspace:
                y_redshift = 3.e-6
            else:
                y_redshift = -0.080
            _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

    # tighten layout, add space for y-label
    _ = fig.tight_layout(pad=0.35)
    _ = plt.subplots_adjust(left=0.1)

    # saving time !
    if args.fname:
        fName = args.fname
    else:
        if args.logspace:
            fName = f'PowerSpectraLogDiff_ALL.png'
        else:
            fName = f'PowerSpectraDiff_ALL.png'
    img_fPath = Path(fName)

    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        outdir_dirPath = Path.cwd()
    img_fPath = outdir_dirPath / img_fPath

    if args.verbose:
        if args.outdir:
            print(f"--- We are placing the output file in : {outdir_dirPath} ---")
        else:
            print(f"--- No output directory specified, so placing image in CWD: {outdir_dirPath} ---")

        if args.fname:
            print(f"--- We are saving the plot with file name : {fName} ---")
        else:
            print(f"--- No output file name specified, so it will be named : {fName} ---")


    _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
    plt.close(fig)

        

if __name__=="__main__":
    main()

