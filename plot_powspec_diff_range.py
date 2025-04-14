#!/usr/bin/env python3
"""
This script will plot the difference of a flux power spectra created from one
    range against the flux power spectra produced On-The-Fly by Cholla. 

Usage of plotting absolute difference:
    $ python3 plot_powspec_diff_range.py 0_fluxpowerspectrum_optdepthbin.h5 0_analysis.h5 -v

Usage of plotting the relative difference:
    $ python3 plot_powspec_diff_rangee.py 0_fluxpowerspectrum_optdepthbin.h5 0_analysis.h5 -v -l
"""
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
    Create a command line argument parser that grabs the FPS optdepthbin file
        and the OTF analysis file. Allow for verbosity

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Plot the flux power spectra from each effective optical depth bin")

    parser.add_argument("FPS_optdepthbin_fname", help='Optical depth binned Flux Power Spectra file', type=str)

    parser.add_argument("nrange", help='Range index within FPS_optdepthbin_fname to use', type=int)

    parser.add_argument("analysis_fname", help='Output analysis file we are plotting against')

    parser.add_argument('-l', '--logspace', help='Display relative difference in log-space',
                        action='store_true')

    parser.add_argument('-f', '--fname', help='Output file name', type=str)

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser



def main():
    '''
    Plot and compare power spectra from opt depth binning against a Cholla OTF analysis file
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at file : {args.FPS_optdepthbin_fname} ---")
        print(f"--- We are looking at range index : {args.nrange} ---")
        print(f"--- We are comparing against : {args.analysis_fname}---")

        if args.outdir:
            print(f"--- We are placing the output files in : {args.outdir} ---")
        else:
            print(f"--- No output directory specified, so placing image in same directory as file ---")

    assert args.nrange >= 0

    precision = np.float64

    FPSoptdepthbin_fPath = Path(args.FPS_optdepthbin_fname).resolve()
    assert FPSoptdepthbin_fPath.is_file()

    analysis_fPath = Path(args.analysis_fname).resolve()
    assert analysis_fPath.is_file()

    # get power spectra from analysis path
    Pk_analysis = np.array([])
    current_z = 0.
    dlogk_analysis = 0.
    with h5py.File(analysis_fPath, 'r') as fObj_analysis:
        Pk_analysis = fObj_analysis['lya_statistics']['power_spectrum'].get('p(k)')[:]
        kvals_analysis = fObj_analysis['lya_statistics']['power_spectrum'].get('k_vals')[:]
        current_z = fObj_analysis.attrs['current_z'].item()

        l_kvals_analysis = np.log10(kvals_analysis)
        dlogk_analysis = np.median(l_kvals_analysis[1:] - l_kvals_analysis[:-1])

    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    nOutput = 0
    with h5py.File(FPSoptdepthbin_fPath, 'r') as fObj:
        quantile_groupkeys = fObj.keys()
        # make sure we're able to look at the nrange requested
        nranges = fObj.attrs['nranges'].item()
        assert args.nrange < nranges

        # save nOutput to place in file name
        nOutput = int(fObj.attrs.get('nOutput'))

        # make sure we can actually compare these two files
        assert 'dlogk' in fObj.attrs
        assert np.isclose(fObj.attrs.get('dlogk'), dlogk_analysis)


        # grab k values
        k_edges = fObj.get('k_edges_dlogk')[:]
        l_kedges = np.log10(k_edges)
        l_kcenters = (l_kedges[1:] + l_kedges[:-1]) / 2.
        k_centers = 10**(l_kcenters)

        if args.verbose:
            print("--- Plotting both relative difference ---")
        
        nrange_key = f'FluxPowerSpectrum_range_{args.nrange:.0f}'
        FPS_range = fObj.get(nrange_key)
        FPS_all = FPS_range.get('FPS_dlogk')[:]
        
        # calculate delta2F and relative difference wrt Cholla OTF analysis 
        delta2F = (1. / np.pi) * k_centers * FPS_all
        delta2F_analysis = (1. / np.pi) * k_centers * Pk_analysis
        delta2F_fracdiff = (delta2F - delta2F_analysis) / delta2F_analysis

        # no nans here !
        goodDelta2F = ~np.isnan(delta2F_fracdiff)

        # use my k_centers to plot
        _ = ax.plot(k_centers[goodDelta2F], np.abs(delta2F_fracdiff[goodDelta2F]))

    # place labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    ylabel_str = r'$D [ \Delta_F^2 (k) ] $'
    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    if args.logspace:
        ylow, yupp = 1e-8, 1e-2
    else:
        ylow, yupp = -0.05, 0.05
    _ = ax.set_ylim(ylow, yupp)

    xlow, xupp = 1e-3, 1e-1
    _ = ax.set_xlim(xlow, xupp)

    # set log-scale
    _ = ax.set_xscale('log')
    if args.logspace:
        _ = ax.set_yscale('log')

    # add redshift str
    xlow, xupp = 1e-3, 5e-2
    redshift_str = rf"$z = {current_z:.4f}$"
    x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
    if args.logspace:
        y_redshift = 3.e-8
    else:
        y_redshift = -0.03
    _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    # define file name of plot
    if args.fname:
        fName = args.fname
    else:
        if args.logspace:
            fName = f"{nOutput:.0f}_FluxPowerSpectra_LogDiff.png"
        else:
            fName = f"{nOutput:.0f}_FluxPowerSpectra_Diff.png"
    img_fPath = Path(fName)

    # define where file name will be placed
    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        # write data to where FPS opt depth bin analysis file resides
        outdir_dirPath = FPSoptdepthbin_fPath.parent.resolve()
    img_fPath = outdir_dirPath / img_fPath

    if args.verbose:
        if args.fname:
            print(f"--- We are saving the plot with name and path : {img_fPath} ---")
        else:
            print(f"--- No output file name specified, so it will be placed as : {img_fPath} ---")

    _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
    plt.close(fig)



if __name__=="__main__":
    main()
