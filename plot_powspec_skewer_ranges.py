#!/usr/bin/env python3
"""
This script will plot the distribution of flux power spectra that was placed
    in a distribution of quantiles. All flux power spectra are shown with the
    color of its line being representative of the mean effective optical
    depth in that quantile.

Usage of saving all individual flux power spectra grouped by unique k values:
    $ python3 plot_powspec_skewer_quantile.py 0_fluxpowerspectrum_optdepthbin.h5 -v -a -u

Usage of saving flux power spectra grouped by dlogk:
    $ python3 plot_powspec_skewer_quantile.py 0_fluxpowerspectrum_optdepthbin.h5 -v -g
"""

import argparse
from pathlib import Path

import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


plt.style.use("dstyle")
_ = plt.figure()

###
# Create command line arg parser
###

def create_parser():
    '''
    Create a command line argument parser that grabs the FPS optdepthbin name.
        Allows for verbosity.

    Args:
        ...
    Returns:
        ...
    '''

    parser = argparse.ArgumentParser(
        description="Plot the flux power spectra from each effective optical depth bin")

    parser.add_argument("FPS_optdepthbin_fname", help='Optical depth binned Flux Power Spectra file', type=str)

    parser.add_argument('-f', '--fname', help='Output file name', type=str)

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-a', '--saveall', help='Whether to save all quantile', 
                        action='store_true')

    parser.add_argument('-u', '--unique', help="Whether to use unique k grouped spectra",
                        action='store_true')

    parser.add_argument('-d', '--dlogk', help='Whether to use dlogk grouped spectra',
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


def main():
    '''
    Plot the flux power spectrum for one skewer from powspec_skewer_quantiles.py
    '''

    # Create parser
    parser = create_parser()

    # Save args
    args = parser.parse_args()

    if args.verbose:
        print("we're verbose in this mf !")
        print(f"--- We are looking at file : {args.FPS_optdepthbin_fname} ---")

        if args.outdir:
            print(f"--- We are placing the output files in : {args.outdir} ---")
        else:
            print(f"--- No output directory specified, so placing image in same directory as file ---")

        if args.saveall:
            print(f"--- We are saving all quantile plots ---")
        else:
            print(f"--- We are only saving the combined plot ---")

        if args.dlogk:
            print("--- Using spectra grouped by dlogk ---")
        elif args.unique:
            print("--- Using spectra grouped by unique k modes---")
        else:
            print("--- Ruh-roh! You need to specify either unique or dlogk flag---")

        if args.dlogk and args.unique:
            print("--- Ruh-roh! You can't specify both unique and dlogk !---")

    # ^ is xor (exclusive or) operator
    assert args.dlogk ^ args.unique

    precision = np.float64

    analysis_fPath = Path(args.FPS_optdepthbin_fname).resolve()
    assert analysis_fPath.is_file()

    # place max number of ranges up to (number of colors in dstyle)
    maxnRange = 7

    # initialize info for plotting
    nlines = 0
    lines_powspec = []
    labels_powspec = []
    redshift = 0.
    nOutput = 0

    with h5py.File(analysis_fPath, 'r') as fObj:
        # grab required info
        nRanges = fObj.attrs['nranges'].item()
        if nRanges <= maxnRange:
            print(f"--- Plotting all {nRanges:.0f} ranges---")
            nRanges2Plot = nRanges
        else:
            print(f"--- Plotting the first {maxnRange:.0f} ranges only !---")
            nRanges2Plot = maxnRange

        nOutput = fObj.attrs.get('nOutput')
        scale_factor = fObj.attrs.get('scale_factor')
        redshift = fObj.attrs.get('redshift')

        if args.verbose:
            print(f"--- Redshift / Scale factor : {redshift:.4f} / {scale_factor:.4f}")
            print(f"--- Number of Ranges : {nRanges:.0f}")
            print(f"--- Original skewer file was output number {nOutput:.0f}")

        # grab k values
        if args.dlogk:
            assert 'dlogk' in fObj.attrs
            k_edges = fObj.get('k_edges_dlogk')[:]
            l_kedges = np.log10(k_edges)
            l_kcenters = (l_kedges[1:] + l_kedges[:-1]) / 2.
            k_centers = 10**(l_kcenters)
        if args.unique:
            assert 'k_uniq' in fObj.keys()
            k_centers = fObj.get('k_uniq')[:]

        if args.verbose:
            curr_str = f'--- Distribution of skewers in nOutput {nOutput} / scale factor: '
            curr_str += f'{scale_factor:.4f} / redshift: {redshift:.4f} --- '
            print(curr_str)
            print(f'--- | range | tau_min | tau_max | Mean tau_eff | ---')


        for nRange in range(nRanges2Plot):
            currRange_key = f'FluxPowerSpectrum_range_{nRange:.0f}'
            FPS_currRange = fObj.get(currRange_key)

            tau_eff_min = FPS_currRange.attrs.get('tau_min')
            tau_eff_max = FPS_currRange.attrs.get('tau_max')
            tau_eff_mean = FPS_currRange.attrs.get('tau_mean')
            
            if args.verbose:
                curr_str = f"--- | {nRange:.0f} | "
                curr_str += f"{tau_eff_min:.4e} | "
                curr_str += f"{tau_eff_max:.4e} | "
                curr_str += f"{tau_eff_mean:.4e} | --- "
                print(curr_str)

            # grab FPS values
            if args.dlogk:
                FPS_all = FPS_currRange.get('FPS_dlogk')[:]
            if args.unique:
                FPS_all = FPS_currRange.get('FPS_uniq')[:]

            # calculate delta2F and clear out zeros & nans
            delta2F = (1. / np.pi) * k_centers * FPS_all
            goodDelta2F_mask = ~((delta2F == 0.) | (np.isnan(delta2F)))
            goodDelta2F = delta2F[goodDelta2F_mask]
            goodkcenters = k_centers[goodDelta2F_mask]

            # save k vs delta2F element
            lines_powspec_elm = np.zeros((goodDelta2F.size, 2))
            lines_powspec_elm[:,0] = goodkcenters
            lines_powspec_elm[:,1] = goodDelta2F
            _ = lines_powspec.append(lines_powspec_elm)

            # save tau min and tau max
            _ = labels_powspec.append((tau_eff_min, tau_eff_max))

            if args.verbose and args.saveall:
                print(f"Appending range {nRange:.0f}")

            nlines += 1



    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

    # add each lines
    for k in range(nlines):
        kcenters, delta2F = lines_powspec[k].T
        tau_min, tau_max = labels_powspec[k]
        label_str = rf"${{{np.log10(tau_min):.2f}}}$" + r"$< \log_{10} \tau_{\rm{eff}} < $" + rf"$ {{{np.log10(tau_max):.2f}}}  $"
        _ = ax.plot(kcenters, delta2F, label=label_str)


    # place labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    ylabel_str = r'$ \Delta_F^2 (k) $'
    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    ylow, yupp = 1.e-3, 1.e1
    _ = ax.set_ylim(ylow, yupp)

    xlow, xupp = 1e-3, 1e-1
    _ = ax.set_xlim(xlow, xupp)

    # set log-scale
    _ = ax.set_xscale('log')
    _ = ax.set_yscale('log')

    # add redshift info
    redshift_str = rf"$z = {{{redshift:.4f}}}$"
    x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
    y_redshift = yupp / 3.
    _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    _ = ax.legend()

    # define file name of plot
    if args.fname:
        fName = args.fname
    else:
        if args.dlogk:
            fName = f"{nOutput:.0f}_FluxPowerSpectra_Ranges_dlogK.png"
        elif args.unique:
            fName = f"{nOutput:.0f}_FluxPowerSpectra_Ranges_UniqueK.png"
    img_fPath = Path(fName)

    # define where file name will be placed
    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        # write data to where FPS opt depth bin analysis file resides
        outdir_dirPath = analysis_fPath.parent.resolve()
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
