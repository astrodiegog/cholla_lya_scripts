#!/usr/bin/env python3
"""
This script groups flux power spectra from each x-y-z axis. This script loops
    over every group in nOutput_fluxpowerspectrum_optdepthbin.h5 and combines
    power spectra by finding the unique k-mode values along each axis.

Usage:
    $ python3 group_powspec_uniquek.py 0_fluxpowerspectrum_optdepthbin.h5 -v
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
        description="Group power spectra by unique k-mode values")

    parser.add_argument("FPS_optdepthbin_fname", help='Optical depth binned Flux Power Spectra file', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser




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


    precision = np.float64

    analysis_fPath = Path(args.FPS_optdepthbin_fname).resolve()
    assert analysis_fPath.is_file()

    if args.verbose:
        print("--- Grabbing required information to rebin power spectra ---")

    # grab required info
    with h5py.File(analysis_fPath, 'r') as fObj:
        # grab FFT info
        k_x = fObj.get('k_x')[:]
        k_y = fObj.get('k_y')[:]
        k_z = fObj.get('k_z')[:]

    # create array for all k-values
    k_all = np.zeros(k_x.size + k_y.size + k_z.size, dtype=precision)
    k_all[ : (k_x.size)] = k_x
    k_all[ (k_x.size) : (k_x.size + k_y.size)] = k_y
    k_all[ (k_x.size + k_y.size) : ] = k_z

    # find all unique k-mode values, how to index into them, and counts of each
    k_uniq, k_uniq_inv, k_uniq_cts = np.unique(k_all, return_inverse=True, return_counts=True)

    if args.verbose:
        print("--- Just figured out how many FFT modes land in each bin. Now writing data ---")

    # create grouped FPS for each quantile
    with h5py.File(analysis_fPath, 'r+') as fObj:
        nQuantiles = fObj.attrs.get('nquantiles')

        # flush out old uniquek analysis
        uniquek_analysis_alive = 'k_uniq' in fObj.keys()
        if uniquek_analysis_alive:
            del fObj['k_uniq']

        # write k modes
        _ = fObj.create_dataset('k_uniq', data=k_uniq)

        for nQuantile in range(nQuantiles):
            currQuantile_key = f'FluxPowerSpectrum_quantile_{nQuantile:.0f}'
            quantile_group = fObj[currQuantile_key]
            # grab FPS arrays
            FPS_x = quantile_group.get('FPS_x')[:]
            FPS_y = quantile_group.get('FPS_y')[:]
            FPS_z = quantile_group.get('FPS_z')[:]

            FPS_currQuantile = np.zeros_like(k_uniq)
            _ = np.add.at(FPS_currQuantile, k_uniq_inv[ : (k_x.size)], FPS_x)
            _ = np.add.at(FPS_currQuantile, k_uniq_inv[(k_x.size) : (k_x.size + k_y.size)], FPS_y)
            _ = np.add.at(FPS_currQuantile, k_uniq_inv[(k_x.size + k_y.size) : ], FPS_z)
            FPS_currQuantile /= k_uniq_cts

            # delete data set if it is alive
            if uniquek_analysis_alive:
                del quantile_group['FPS_uniq']
        
            # write data
            _ = quantile_group.create_dataset('FPS_uniq', data=FPS_currQuantile)
    


    if args.verbose:
        print("--- Done ! ---")
 


if __name__=="__main__":
    main()


