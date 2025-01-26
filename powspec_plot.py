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

    parser.add_argument("skewfname", help='Cholla skewer output file name', type=str)

    parser.add_argument("analysisfname", help='Cholla analysis output file name', type=str)

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
    assert len(Pk_avgs) == len(labels)
    # make sure all power spectra are of the same size
    for Pk_avg in Pk_avgs:
        assert k_model.size == Pk_avg.size


    for i, Pk_avg in enumerate(Pk_avgs):
        label = labels[i]
        delta2F = (1. / np.pi) * k_model * Pk_avg

        # no nans here !
        goodDelta2F = (~np.isnan(delta2F)) & (~(delta2F == 0))
        _ = ax.plot(k_model[goodDelta2F], delta2F[goodDelta2F], label=label)

    # plase labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    ylabel_str = r'$ \Delta_F^2 (k) $'

    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    ylow, yupp = 1e-4, 1e0
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
        ax (Axes.Image?):
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

    # plase labels & limits if not already set
    xlabel_str = r'$k\ [\rm{s\ km^{-1}}] $'
    if log:
        ylabel_str = r'$| D [ \Delta_F^2 (k) ] |$'
    else:
        ylabel_str = r'$D [ \Delta_F^2 (k) ] $'

    _ = ax.set_xlabel(xlabel_str)
    _ = ax.set_ylabel(ylabel_str)

    if log:
        ylow, yupp = 1e-8, 1e-2
        _ = ax.set_ylim(ylow, yupp)
        _ = ax.set_yscale('log')
    else:
        ylow, yupp = -0.01, 0.01
        _ = ax.set_ylim(ylow, yupp)

    # add x lims
    xlow, xupp = 1e-3, 5e-2
    _ = ax.set_xlim(xlow, xupp)

    # set x log-scale
    _ = ax.set_xscale('log')

    # add background grid
    _ = ax.grid(which='both', axis='both', alpha=0.3)

    _ = ax.legend(fontsize=14, loc='upper right')

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
        print(f"--- Comparing against analysis file : {args.analysisfname} ---")
        if args.difference:
            print(f"--- We are displaying the difference ---")
        if args.logspace:
            print(f"--- We are showing data in log-space ---")
            if not args.difference:
                print(f"--- I mean.... already planning to show flux power spectra in logspace... ty? ---")

    if not args.difference:
        args.logspace = True

    precision = np.float64

    skewer_fPath = Path(args.skewfname).resolve()
    analysis_fPath = Path(args.analysisfname).resolve()
    assert skewer_fPath.is_file()
    assert analysis_fPath.is_file()    

    # get power spectra from analysis path
    k_skew = np.array([])
    Pk_analysis = np.array([])
    current_z_analysis = 0.
    with h5py.File(analysis_fPath, 'r') as fObj_analysis:
        Pk_analysis = fObj_analysis['lya_statistics']['power_spectrum'].get('p(k)')[:]
        k_skew = fObj_analysis['lya_statistics']['power_spectrum'].get('k_vals')[:]
        current_z_analysis = fObj_analysis.attrs['current_z'].item()

    # get power spectra we calculated
    Pk_avg = np.array([])
    Pk_avg_newDeltaFCalc = np.array([])
    current_z = 0.
    with h5py.File(skewer_fPath, 'r') as fObj_skewer:
        Pk_avg = fObj_skewer['PowerSpectrum'].get('P(k)')[:] 
        Pk_avg_newDeltaFCalc = fObj_skewer['PowerSpectrum_newDeltaFCalc'].get('P(k)')[:]
        current_z = fObj_skewer.attrs['current_z'].item()
    
    # make sure analysis and skewer were output at the same redshift
    assert current_z == current_z_analysis
    
    # plottin time !
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    if args.difference:
        Pk_avg_tests = [Pk_avg, Pk_avg_newDeltaFCalc]
        testlabels = ['oldDeltaFCalc', 'newDeltaFCalc']
        plotFluxPowerSpectra_RelDiff(ax, k_skew, Pk_analysis, Pk_avg_tests, testlabels, args.logspace)
    else:
        Pk_avgs = [Pk_analysis, Pk_avg, Pk_avg_newDeltaFCalc]
        labels = ['chollaDeltaFCalc', 'oldDeltaFCalc', 'newDeltaFCalc']
        plotFluxPowerSpectra(ax, k_skew, Pk_avgs, labels)

    # add redshift str
    xlow, xupp = 1e-3, 5e-2
    redshift_str = rf"$z = {current_z:.4f}$"
    x_redshift = 10**(np.log10(xlow) + (0.05 * (np.log10(xupp) - np.log10(xlow))))
    if args.difference:
        if args.logspace:
            y_redshift = 3.e-8
        else:
            y_redshift = -0.080
    else:
        y_redshift = 3.e-4

    _ = ax.annotate(redshift_str, xy=(x_redshift, y_redshift), fontsize=20)
    
    # tighten layout
    _ = fig.tight_layout()

    # saving time !
    if args.fname:
        fName = args.fname
    else:
        if args.difference:
            if args.logspace:
                fName = f'PowerSpectraLogDiff_z{current_z:.4f}_.png'
            else:
                fName = f'PowerSpectraDiff_z{current_z:.4f}_.png'
        else:
            fname = f'PowerSpectra_z{current_z:.4f}_.png'
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

