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
