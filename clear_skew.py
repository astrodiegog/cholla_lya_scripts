import argparse
from pathlib import Path

import numpy as np
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
        description="Compute and append optical depth")

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser

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

def clear_taucalc(OTFSkewers, verbose=False):
    '''
    Clearing the calculation of the effective optical depth. For each skewers_i axis
        group, remove all datasets created from init_taucalc. Reset back to Cholla
        outputs only

    Args:
        OTFSkewers (ChollaOnTheFlySkewers): holds OTF skewers specific info
        verbose (bool): (optional) whether to print important information
    Returns:
        ...

    '''

    attrs2remove = ['taucalc_prog']
    datasets2remove = ['taucalc_bool', 'taucalc_eff_1sig', 'taucalc_eff_3sig', 'taucalc_eff_5sig',
                        'taucalc_eff_8sig', 'taucalc_eff_10sig', 'taucalc_eff_12sig',
                        'taucalc_eff_allLOS', 'taucalc_local_1sig', 'taucalc_local_3sig', 'taucalc_local_5sig',
                        'taucalc_local_8sig', 'taucalc_local_10sig', 'taucalc_local_12sig',
                        'taucalc_local_allLOS', 'taucalc_time_1sig', 'taucalc_time_3sig', 'taucalc_time_5sig',
                        'taucalc_time_8sig', 'taucalc_time_10sig', 'taucalc_time_12sig',
                        'taucalc_time_allLOS']

    with h5py.File(OTFSkewers.OTFSkewersfPath, 'r+') as fObj:
        if verbose:
            print(f'\t...removing optical depth info for file {OTFSkewers.OTFSkewersfPath}')

        OTFSkewers_lst = [OTFSkewers.get_skewersx_obj(),
                          OTFSkewers.get_skewersy_obj(),
                          OTFSkewers.get_skewersz_obj()]

        # add progress attribute, boolean mask for whether tau is calculated, and tau itself
        for i, OTFSkewers_i in enumerate(OTFSkewers_lst):
            if verbose:
                print(f"\t\t...removing arrays and attributes along axis {i:.0f}")
            skew_key = OTFSkewers_i.OTFSkewersiHead.skew_key

            for attr in attrs2remove:
                if attr in dict(fObj[skew_key].attrs).keys():
                    fObj[skew_key].attrs.pop(attr)
                    if verbose:
                        print(f'\t\t\t... removed attribute {attr}')
                else:
                    if verbose:
                        print(f'\t\t\t... attribute {attr} not found')

            for datakey in datasets2remove:
                if datakey in fObj[skew_key].keys():
                    del fObj[skew_key][datakey]
                    if verbose:
                        print(f'\t\t\t... deleted dataset {datakey}')
                else:
                    if verbose:
                        print(f'\t\t\t... dataset {datakey} not found')

    if verbose:
        print("...clearing complete !")
    
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

    precision = np.float64

    # Convert argument input to Path() & get its absolute path
    skewer_fPath = Path(args.skewfname).resolve()
    assert skewer_fPath.is_file()

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(skewer_fPath)

    # remove progress attribute and all new datasets
    clear_taucalc(OTFSkewers, args.verbose)

        

if __name__=="__main__":
    main()

