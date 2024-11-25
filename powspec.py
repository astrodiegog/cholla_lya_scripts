import argparse
import os

import numpy as np
from scipy.special import erf
import h5py

from cholla_api.analysis.ChollaCosmoCalculator import ChollaCosmologyHead, ChollaSnapCosmologyHead
from cholla_api.snap.ChollaSnap import ChollaSnapHead

from cholla_api.OTFanalysis.ChollaOnTheFlyAnalysis import ChollaOnTheFlyPowerSpectrumHead, ChollaOnTheFlyPowerSpectrum
from cholla_api.OTFanalysis.ChollaOnTheFlySkewers import ChollaOnTheFlySkewers


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

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument('-l', '--local', help='Whether to store local optical depths',
                        action='store_true')

    parser.add_argument('-v', '--verbose', help='Print info along the way',
                        action='store_true')

    return parser


def P_k_i(OTFSkewers_i, dvHubble, dlogk, precision=np.float64):
    '''
    return the total power spectrum along i-axis direction
    '''

    # grab local optical depths
    local_opticaldepth_key = 'taucalc_local'
    local_opticaldepth = np.zeros((OTFSkewers_i.OTFSkewersiHead.n_skews, OTFSkewers_i.OTFSkewersiHead.n_i), dtype=precision)
    fObj = h5py.File(OTFSkewers_i.fPath, 'r')
    local_opticaldepth[:,:] = fObj[OTFSkewers_i.OTFSkewersiHead.skew_key].get(local_opticaldepth_key)[:, :]
    fObj.close()

    # calculate local transmitted flux (& its mean)
    fluxes = np.exp(-local_opticaldepth)
    flux_mean = np.mean(fluxes)

    # find the indices that describe where the k-mode FFT bins fall within kval_edges (from dlogk)
    nfft = int(OTFSkewers_i.OTFSkewersiHead.n_i / 2 + 1)
    OTFPSHead = ChollaOnTheFlyPowerSpectrumHead(dlogk, nfft, OTFSkewers_i.OTFSkewersiHead.n_i, dvHubble)
    OTFPowerSpectrum = ChollaOnTheFlyPowerSpectrum(OTFPSHead, '')
    fft_binids = OTFPowerSpectrum.get_fft_binids(precision, precision)

    # find number of fft modes that fall in requested dlogk bin id (used later to average total power in dlogk bin)
    hist_n = np.zeros(OTFPowerSpectrum.OTFPowerSpectrumHead.n_bins, dtype=precision)
    for bin_id in fft_binids[1:]:
        hist_n[bin_id] += 1.
    # (protect against dividing by zero)
    hist_n[hist_n == 0] = 1.
    
    # calculate Hubble flow across entire box (max velocity)
    umax = OTFPowerSpectrum.OTFPowerSpectrumHead.dvHubble * OTFPowerSpectrum.OTFPowerSpectrumHead.n_los
    
    # initialize total power array
    P_k_tot = np.zeros(OTFPowerSpectrum.OTFPowerSpectrumHead.n_bins, dtype=precision)

    for nSkewerID in range(OTFSkewers_i.OTFSkewersiHead.n_skews):
        # calculate flux fluctuation 
        dFlux_skew = fluxes[nSkewerID] / flux_mean
        
        # calculate fft & amplitude of fft
        fft = np.fft.rfft(dFlux_skew)
        fft2 = (fft.imag * fft.imag + fft.real * fft.real) / OTFPowerSpectrum.OTFPowerSpectrumHead.n_los / OTFPowerSpectrum.OTFPowerSpectrumHead.n_los

        # add power for each fft mode
        hist_PS_vals = np.zeros(OTFPowerSpectrum.OTFPowerSpectrumHead.n_bins, dtype=precision)
        hist_PS_vals[fft_binids[1:]] += fft2[1:]

        # take avg & scale by umax
        delta_F_avg = hist_PS_vals / hist_n
        P_k = umax * delta_F_avg
        P_k_tot += P_k

    # average out the number of skewers
    P_k_mean = P_k_tot / OTFSkewers_i.OTFSkewersiHead.n_skews

    # grab k-mode bin edges
    kmode_edges = OTFPowerSpectrum.get_kvals_edges(precision)
    
    return (kmode_edges, P_k_mean)


def P_k_calc(OTFSkewers, dlogk, save_Pki=False, precision=np.float64):
    '''
    '''
    # create cosmology header
    chCosmoHead = ChollaCosmologyHead(OTFSkewers.Omega_M, OTFSkewers.Omega_R,
                                      OTFSkewers.Omega_K, OTFSkewers.Omega_L,
                                      OTFSkewers.w0, OTFSkewers.wa, OTFSkewers.H0)

    # create skew cosmo calc object
    snapHead = ChollaSnapHead(nSkewerOutput + 1) # snapshots are index-1
    snapHead.a = OTFSkewers.current_a

    # calculate dvHubble in each direction
    snapCosmoHead = ChollaSnapCosmologyHead(snapHead, cosmoHead)
    dvHubble_x = dvHubble(OTFSkewers.dx)
    dvHubble_y = dvHubble(OTFSkewers.dy)
    dvHubble_z = dvHubble(OTFSkewers.dz)

    # calculate power spectra in each direction
    OTFSkewers_x = OTFSkewers.get_skewersx_obj()
    k_x, P_k_x = P_k_i(OTFSkewers_x, dvHubble_x, dlogk, precision)

    OTFSkewers_y = OTFSkewers.get_skewersy_obj()
    k_y, P_k_y = P_k_i(OTFSkewers_y, dvHubble_y, dlogk, precision)

    OTFSkewers_z = OTFSkewers.get_skewersz_obj()
    k_z, P_k_z = P_k_i(OTFSkewers_z, dvHubble_z, dlogk, precision)

    # before combining power spectrum along each direction, make sure they're of same shape
    assert np.array_equal(P_k_x.shape, P_k_y.shape)
    assert np.array_equal(P_k_x.shape, P_k_z.shape)

    # also need to assert that k_xyz are all within some tolerance level
    # for now assume they're the same, and save k_x by default !

    # combined power spectrum
    P_k = (P_k_x + P_k_y + P_k_z) / 3.
    PS_group_key = 'PowerSpectrum'
    PS_avg_key = 'P(k)'
    k_edges_key = 'k_edges'
    with h5py.File(OTFSkewers.fPath, 'r+') as fObj:
        fObj.create_group(PS_group_key)
        fObj[PS_group_key].create_dataset(PS_avg_key, data=P_k)
        fObj[PS_group_key].create_dataset(k_edges_key, data=k_x)
        if save_Pki:
            PS_x_avg_key = 'P_x(k_x)'
            k_x_edges_key = 'k_x_edges'
            PS_y_avg_key = 'P_y(k_y)'
            k_y_edges_key = 'k_y_edges'
            PS_z_avg_key = 'P_z(k_z)'
            k_z_edges_key = 'k_z_edges'
            fObj[PS_group_key].create_dataset(k_x_edges_key, data=k_x)
            fObj[PS_group_key].create_dataset(PS_x_avg_key, data=P_k_x)
            fObj[PS_group_key].create_dataset(k_y_edges_key, data=k_y)
            fObj[PS_group_key].create_dataset(PS_y_avg_key, data=P_k_y)
            fObj[PS_group_key].create_dataset(k_z_edges_key, data=k_z)
            fObj[PS_group_key].create_dataset(PS_z_avg_key, data=P_k_z)

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
        print(f"--- We are looking at skewer file : {args.skewfname} ---")
        print(f"--- We are placing power spectra in dlogk : {args.dlogk} ---")
        if args.save_Pki:
            print(f"--- We are saving power spectra in each direction (!) ---")
        else:
            print(f"--- We are NOT saving power spectra in each direction (!) ---")

    precision = np.float64

    # ensure that local optical depth is a dataset
    fObj = h5py.File(args.skewfname, 'r')
    local_opticaldepth_key = 'taucalc_local'
    assert local_opticaldepth_key in fObj['skewers_x'].keys()
    assert local_opticaldepth_key in fObj['skewers_y'].keys()
    assert local_opticaldepth_key in fObj['skewers_z'].keys()
    fObj.close()

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    # convert relative path to skewer file name to absolute file path
    cwd = os.getcwd()
    if args.skewfname[0] != '/':
        relative_path = args.skewfname
        args.skewfname = cwd + '/' + relative_path

    # seperate the skewer output number and skewer directory
    skewfName = args.skewfname.split('/')[-1]
    nSkewerOutput = int(skewfName.split('_')[0])
    skewersdir = args.skewfname[:-(len(skewfName)+1)]

    # create ChollaOTFSkewers object
    OTFSkewers = ChollaOnTheFlySkewers(nSkewerOutput, skewersdir)

    # calculate the power spectra
    P_k_calc(OTFSkewers, args.dlogk, args.save_Pki, precision=np.float64)


    # maybe in the future I could also save the local power spectra for each skewer?



if __name__="__main__":
    main()


