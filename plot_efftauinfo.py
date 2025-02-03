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
        description="Compute and append power spectra")

    parser.add_argument("skewdirname", help='Cholla skewer output directory name', type=str)

    parser.add_argument("nOutputsStr", help='String of outputs delimited by comma', type=str)

    parser.add_argument('-f', '--fname', help='Output file name', type=str)

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser

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
    Compute the power spectrum and append to skewer file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at skewer directory : {args.skewdirname} ---")

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

    # make sure required keys are populated
    optdeptheff_key = 'taucalc_eff'
    precision = np.float64
    redshift_arr = np.zeros(nOutputs.size, dtype=np.float64)
    redshift_bins = np.zeros(nOutputs.size+1, dtype=np.float64)

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

        assert OTFSkewers_x.check_datakey(optdeptheff_key)
        assert OTFSkewers_y.check_datakey(optdeptheff_key)
        assert OTFSkewers_z.check_datakey(optdeptheff_key)      

        redshift_arr[n] = OTFSkewers.current_z 
        redshift_bins[n] = OTFSkewers.current_z

    # optical depth histogram bins
    l_tau_min, l_tau_max = -3., 2.
    tau_nbins = 250
    l_tau_bins = np.linspace(l_tau_min, l_tau_max, tau_nbins) 

    hist_eff_ltau = np.zeros((tau_nbins-1, nOutputs.size))

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

        # grab effective optical depth, histogram it, place onto global array
        xefftaus = OTFSkewers_x.get_skeweralldata(optdeptheff_key, dtype=precision)
        yefftaus = OTFSkewers_y.get_skeweralldata(optdeptheff_key, dtype=precision)
        zefftaus = OTFSkewers_z.get_skeweralldata(optdeptheff_key, dtype=precision)

        xeff_ltau_hist, _ = np.histogram(np.log10(xefftaus.flatten()), bins=l_tau_bins)
        yeff_ltau_hist, _ = np.histogram(np.log10(yefftaus.flatten()), bins=l_tau_bins)
        zeff_ltau_hist, _ = np.histogram(np.log10(zefftaus.flatten()), bins=l_tau_bins)

        hist_eff_ltau[:,nOutput] += xeff_ltau_hist
        hist_eff_ltau[:,nOutput] += yeff_ltau_hist
        hist_eff_ltau[:,nOutput] += zeff_ltau_hist

        xnskews = xefftaus.size
        ynskews = yefftaus.size
        znskews = zefftaus.size
        totnskews = xnskews + ynskews + znskews

        normbynskews = True
        if normbynskews:
            # normalize by number of skewers
            norm_const = totnskews
        else:
            # normalize by the total makeup in the bins
            norm_const = np.sum(hist_eff_ltau[:,nOutput])

        # normalize at each redshift bin
        hist_eff_ltau[:,nOutput] = hist_eff_ltau[:,nOutput] / norm_const

    yaxis_label = r'$\tau_{\rm{med}}$'
    yaxis_low, yaxis_hi = 1.e-3, 1.e2
    xlabel_str = r'$z$'
    z_low, z_hi  0., 8.5

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    im0 = ax.pcolormesh(redshift_bins, 10**(l_tau_bins), np.log10(hist_eff_ltau))

    _ = ax.set_xlim(z_low, z_hi)
    _ = ax.set_ylim(yaxis_low, yaxis_hi)
    _ = ax.set_yscale('log')

    # add background grid and legend
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    # place plot label
    x_plotlabel = z_low + (0.05 * (z_hi - z_low))
    y_plotlabel = 10**(np.log10(yaxis_low) + (0.85 * (np.log10(yaxis_hi) - np.log10(yaxis_low))))
    _ = ax.annotate(plotlabel, xy=(x_plotlabel, y_plotlabel), fontsize=20)

    # place colorbar
    cbar_ax = fig_efftau.add_axes([0.515, 0.142 , 0.14, 0.015])
    _ = fig_efftau.colorbar(im0, cax=cbar_ax, orientation="horizontal")
    _ = cbar_ax.yaxis.set_ticks_position('right')

    # add colorbar label & ensure no overlap w/ticks
    cbar_str = r"$\log_{10} \rm{P}(\tau_{\rm{med}} | z)$"
    _ = cbar_ax.set_xlabel(cbar_str)
    _ = cbar_ax.xaxis.set_label_position('top')
    _ = cbar_ax.xaxis.labelpad = 5

    # place redshift label
    _ = ax.set_xlabel(xlabel_str)

    # place yaxis
    _ = ax.set_ylabel(yaxis_label)

    _ = fig.tight_layout()

    fig_efftau.savefig(efftau_fImgPath, dpi=256, bbox_inches='tight')

if __name__=="__main__":
    main()
