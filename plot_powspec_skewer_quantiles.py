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

    if args.outdir:
        outdir_dirPath = Path(args.outdir)
        outdir_dirPath = outdir_dirPath.resolve()
        assert outdir_dirPath.is_dir()
    else:
        # write data to where skewer directory resides
        outdir_dirPath = analysis_fPath.parent.resolve()


    # initialize info for plotting
    lines_powspec = []
    redshift = 0.
    nOutput = 0

    with h5py.File(analysis_fPath, 'r') as fObj:
        # grab required info
        nQuantiles = fObj.attrs.get('nquantiles')
        nOutput = fObj.attrs.get('nOutput')
        scale_factor = fObj.attrs.get('scale_factor')
        redshift = fObj.attrs.get('redshift')

        if args.verbose:
            print(f"--- Redshift / Scale factor : {redshift:.4f} / {scale_factor:.4f}")
            print(f"--- Number of Quantiles : {nQuantiles:.0f}")
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

        # save log_10 of all median optical depths in each quantile
        l_optdepth_mean = np.zeros(nQuantiles)

        if args.verbose:
            curr_str = f'--- Distribution of skewers in nOutput {nOutput} / scale factor: '
            curr_str += f'{scale_factor:.4f} / redshift: {redshift:.4f} --- '
            print(curr_str)
            print(f'--- | nquantile | tau_min | tau_max | Mean tau_eff | ---')

        for nQuantile in range(nQuantiles):
            currQuantile_key = f'FluxPowerSpectrum_quantile_{nQuantile:.0f}'
            FPS_currQuantile = fObj.get(currQuantile_key)
            tau_eff_min = FPS_currQuantile.attrs.get('tau_min')
            tau_eff_max = FPS_currQuantile.attrs.get('tau_max')
            tau_eff_mean = FPS_currQuantile.attrs.get('tau_mean')
            l_optdepth_mean[nQuantile] = np.log10(tau_eff_mean)
            
            if args.verbose:
                curr_str = f"--- | {nQuantile:.0f} | "
                curr_str += f"{tau_eff_min:.4e} | "
                curr_str += f"{tau_eff_max:.4e} | "
                curr_str += f"{tau_eff_mean:.4e} % | --- "
                print(curr_str)

            # grab FPS values
            if args.dlogk:
                FPS_all = FPS_currQuantile.get('FPS_dlogk')[:]
            if args.unique:
                FPS_all = FPS_currQuantile.get('FPS_uniq')[:]

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

            if args.verbose and args.saveall:
                print(f"Plotting quantile {nQuantile:.0f}")

            if args.saveall:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
                _ = ax.plot(goodkcenters, goodDelta2F)

                # place labels & limits
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

                # add background grid
                _ = ax.grid(which='both', axis='both', alpha=0.3)

                # add optical depth info
                taueff_str = rf"${tau_eff_min:.4f} <$"
                taueff_str += r"$\tau_{ \rm{eff} }$"
                taueff_str += rf"$< {tau_eff_max:.4f}$"
                x_taueff = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
                y_taueff = ylow * 3.
                _ = ax.annotate(taueff_str, xy=(x_taueff, y_taueff), fontsize=20)

                # add redshift info
                redshift_str = rf"$z = {redshift:.3f}$"
                x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
                y_redshift = yupp / 3.
                _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

                # save figure
                fName = f"{nOutput:.0f}_{currQuantile_key}.png"
                img_fPath = Path(fName)
                img_fPath = outdir_dirPath / img_fPath
                if args.verbose:
                    print(f"--- Saving plot {img_fPath} ---")
    
                _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
                plt.close(fig)

    # combine power spectra to a LineCollection object
    linescollec_powspec = LineCollection(lines_powspec, array=l_optdepth_mean, cmap='rainbow', alpha=0.7)

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

    xlow, xupp = 1e-3, 1e-1
    _ = ax.set_xlim(xlow, xupp)

    # set log-scale
    _ = ax.set_xscale('log')
    _ = ax.set_yscale('log')

    # add redshift info
    redshift_str = rf"$z = {redshift:.3f}$"
    x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
    y_redshift = yupp / 3.
    _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

    # place colorbar
    cbar_ax = fig.add_axes([0.93, 0.106, 0.04, 0.78])
    _ = fig.colorbar(linescollec_powspec, cax=cbar_ax, orientation="vertical")
    _ = cbar_ax.yaxis.set_ticks_position('right')

    # add colorbar label & ensure no overlap w/ticks
    cbar_str = r"$\log_{10} \overline{\tau_{\rm{eff}}}$"
    _ = cbar_ax.set_ylabel(cbar_str, rotation=270)
    _ = cbar_ax.yaxis.set_label_position('right')
    _ = cbar_ax.yaxis.labelpad = 20

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    # save figure
    fName = f"{nOutput:.0f}_FluxPowerSpectra_optdepthbin.png"
    img_fPath = Path(fName)
    img_fPath = outdir_dirPath / img_fPath
    if args.verbose:
        print(f"--- Saving Flux Power Spectra plot {img_fPath} ---")

    _ = fig.savefig(img_fPath, dpi=256, bbox_inches = "tight")
    plt.close(fig)


if __name__=="__main__":
    main()
