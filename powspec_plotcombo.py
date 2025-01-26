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

    parser.add_argument('-d', '--difference', help='Display relative difference',
                        action='store_true')

    parser.add_argument('-l', '--logspace', help='Display relative difference in log-space',
                        action='store_true')   
 
    parser.add_argument('-f', '--fname', help='Output file name', type=str)

    parser.add_argument('-o', '--outdir', help='Output directory', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    return parser




###
# Study specific functions
###

def plotFluxPowerSpectra(ax, k_model, Pk_avgs, labels):
    '''
    Plot the relative difference of a set of flux power spectra
    
    Args:
        ax (Axes.Image?):
        k_model (arr): k-centers of the flux power spectrum
        Pk_avgs (list of arrs): arrays of flux power spectra
        labels (list of strs): strs to label each flux power spectra
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

    for i, Pk_avg in enumerate(Pk_avgs):
        label = labels[i]
        delta2F = (1. / np.pi) * k_model * Pk_avg

        # no nans here !
        goodDelta2F = ~np.isnan(delta2F)
        _ = ax.plot(k_model[goodDelta2F], delta2F_fracdiff[goodDelta2F], label=label)

    # plase labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    ylabel_str = r'$ \Delta_F^2 (k) $'

    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    ylow, yupp = 1e-3, 1e0
    _ = ax.set_ylim(ylow, yupp)
    _ = ax.set_yscale('log')

    # add x lims
    xlow, xupp = 1e-3, 5e-2
    _ = ax.set_xlim(xlow, xupp)

    # set x log-scale
    _ = ax.set_xscale('log')

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    _ = ax.legend(fontsize=14, loc='upper right')

    return

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

            ax = ax_all[i][j]    

            # get power spectra from analysis path
            k_skew = np.array(_)
            Pk_analysis = np.array(_)
            current_z_analysis = 0.
            with h5py.File(analysis_fPath, 'r') as fObj_analysis:
                Pk_analysis = fObj_analysis['lya_statistics']['power_spectrum'].get('p(k)')[:]
                k_skew = fObj_analysis['lya_statistics']['power_spectrum'].get('k_vals')[:]
                current_z_analysis = fObj_analysis.attrs['current_z'].item()

            # get power spectra we calculated
            Pk_avg = np.array(_)
            Pk_avg_newDeltaFCalc = np.array(_)
            current_z = 0.
            with h5py.File(skewer_fPath, 'r') as fObj_skewer:
                Pk_avg = fObj_skewer['PowerSpectrum'].get('P(k)')[:]
                Pk_avg_newDeltaFCalc = fObj_skewer['PowerSpectrum_newDeltaFCalc'].get('P(k)')[:]
                current_z = fObj_skewer.attrs['current_z'].item()

            # make sure analysis and skewer were output at the same redshift
            assert current_z == current_z_analysis

            if args.difference:
                Pk_avg_tests = [Pk_avg, Pk_avg_newDeltaFCalc]
                testlabels = ['oldDeltaFCalc', 'newDeltaFCalc']
                plotFluxPowerSpectra_RelDiff(ax, k_skew, Pk_analysis, Pk_avg_tests, testlabels, args.logspace)
            else:
                Pk_avgs = [Pk_analysis, Pk_avg, Pk_avg_newDeltaFCalc]
                labels = ['chollaDeltaFCalc', 'oldDeltaFCalc', 'newDeltaFCalc']
                plotFluxPowerSpectra(ax, k_skew, Pk_avgs, labels)

            # place labels
            if (i == 3):
                xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
            else:
                xlabel_str = None
                _ = ax.tick_params(labelbottom=False)
            if (j == 0):
                if args.differense:
                    if args.logspace:
                        ylabel_str = r'$| D [ \Delta_F^2 (k) ] |$'
                    else:
                        ylabel_str = r'$D [ \Delta_F^2 (k) ] $'
                else:
                    ylabel_str = r'$\Delta_F^2 (k) $'
            else:
                ylabel_str = None
                _ = ax.tick_params(labelleft=False)

            _ = ax.set_xlabel(xlabel_str)
            _ = ax.set_ylabel(ylabel_str)

            # add redshift str
            xlow, xupp = 1e-3, 5e-2
            redshift_str = rf"$z = {current_z:.4f}$"
            x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
            if args.difference:
                if args.logspace:
                    y_redshift = 1.e-6
                else:
                    y_redshift = -0.080
            else:
                y_redshift = 1.e-3
            _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)

    # tighten layout, add space for y-label
    _ = fig.tight_layout(pad=0.35)
    _ = plt.subplots_adjust(left=0.1)

    # saving time !
    if args.fname:
        fName = args.fname
    else:
        if args.difference:
            if args.logspace:
                fName = f'PowerSpectraLogDiff_combo.png'
            else:
                fName = f'PowerSpectraDiff_combo.png'
        else:
            fname = f'PowerSpectra_combo.png'
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

