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
    Create a command line argument parser that grabs the number of nodes
        and the parameter text file. Allow for verbosity

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

    parser.add_argument('-a', '--saveall', help='Whether to save all quantile', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser






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
        print(f"--- We are looking at file : {args.FPS_optdepthbin_fname} ---")

        if args.outdir:
            print(f"--- We are placing the output files in : {args.outdir} ---")
        else:
            print(f"--- No output directory specified, so placing images in CWD ---")

        if args.saveall:
            print(f"--- We are saving all quantile plots ---")
        else:
            print(f"--- We are only saving the combined plot ---")

    precision = np.float64

    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        outdir_dirPath = Path.cwd()

    analysis_fPath = Path(args.FPS_optdepthbin_fname).resolve()

    optdepth_mean = np.array([])
    lines_powspec = []

    with h5py.File(analysis_fPath, 'r') as fObj:
        quantile_groupkeys = fObj.keys()
        k_edges = fObj.attrs['k_edges']
        nQuantiles = fObj.attrs['nquantiles']
        l_kedges = np.log10(k_edges)
        d_l_kedges = l_kedges[1:] - l_kedges[:-1]
        l_kcenters = l_kedges[:-1]  + (d_l_kedges / 2.)
        k_centers = 10**(l_kcenters)
        l_optdepth_mean = np.zeros(nQuantiles)

        for nQuantile, quantile_groupkey in enumerate(quantile_groupkeys):
            FPS_currQuantile = fObj[quantile_groupkey]['P(k)']
            optdepth_min, optdepth_max = fObj[quantile_groupkey].attrs['tau_min'], fObj[quantile_groupkey].attrs['tau_max']
            delta2F = (1. / np.pi) * k_centers * FPS_currQuantile
            goodDelta2F_mask = ~((delta2F == 0.) | (np.isnan(delta2F)))
            goodDelta2F = delta2F[goodDelta2F_mask]
            goodkcenters = k_centers[goodDelta2F_mask]

            lookback_elm = np.zeros((np.sum(goodDelta2F_mask), 2))
            lookback_elm[:,0] = goodkcenters
            lookback_elm[:,1] = goodDelta2F

            _ = lines_powspec.append(lookback_elm)
            l_optdepth_mean[nQuantile] = np.log10(fObj[quantile_groupkey].attrs['tau_mean'])

            if args.verbose and args.saveall:
                print(f"Plotting quantile {nQuantile:.0f}")

            if args.saveall:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                _ = ax.plot(goodkcenters, goodDelta2F)

                # place labels & limits if not already set
                xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
                ylabel_str = r'$ \Delta_F^2 (k) $'
                _ = ax.set_xlabel(xlabel_str)
                _ = ax.set_ylabel(ylabel_str)

                ylow, yupp = 1.e-3, 1.e1
                _ = ax.set_ylim(ylow, yupp)

                xlow, xupp = 1e-3, 5e-2
                _ = ax.set_xlim(xlow, xupp)

                # set log-scale
                _ = ax.set_xscale('log')
                _ = ax.set_yscale('log')

                # add background grid
                _ = ax.grid(which='both', axis='both', alpha=0.3)

                # add optical depth info
                optdepth_str = rf"${optdepth_min:.4f} < \tau < {optdepth_max:.4f}$"
                x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
                y_redshift = ylow * 3.
                _ = ax.annotate(optdepth_str, xy=(x_redshift, y_redshift), fontsize=20)

                # save figure
                fName = f"{quantile_groupkey}.png"
                img_fPath = Path(fName)
                img_fPath = outdir_dirPath / img_fPath
    
                _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
                plt.close(fig)


    # combine power spectra to a LineCollection object
    linescollec_powspec = LineCollection(lines_powspec, array=l_optdepth_mean, cmap='rainbow')

    if args.verbose:
        print("Plotting combined Flux Power Spectra")
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    # add line collection to plot
    _ = ax.add_collection(linescollec_powspec)

    # place labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    ylabel_str = r'$ \Delta_F^2 (k) $'
    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    ylow, yupp = 1.e-3, 1.e1
    _ = ax.set_ylim(ylow, yupp)

    xlow, xupp = 1e-3, 5e-2
    _ = ax.set_xlim(xlow, xupp)

    # set log-scale
    _ = ax.set_xscale('log')
    _ = ax.set_yscale('log')

    # place colorbar
    cbar_ax = fig.add_axes([0.93, 0.106, 0.04, 0.78])
    _ = fig.colorbar(linescollec_powspec, cax=cbar_ax, orientation="vertical")
    _ = cbar_ax.yaxis.set_ticks_position('right')

    # add colorbar label & ensure no overlap w/ticks
    cbar_str = r"$\log_{10} \tau_{\rm{mean}}$"
    _ = cbar_ax.set_ylabel(cbar_str, rotation=270)
    _ = cbar_ax.yaxis.set_label_position('right')
    _ = cbar_ax.yaxis.labelpad = 20

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    # save figure
    fName = f"FluxPowerSpectra_optdepthbin.png"
    img_fPath = Path(fName)
    img_fPath = outdir_dirPath / img_fPath

    _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
    plt.close(fig)



if __name__=="__main__":
    main()
