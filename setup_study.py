import argparse
from pathlib import Path

import numpy as np

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
        description="Setup study")

    parser.add_argument("skewDir", help='Cholla skewer output directory name', type=str)

    parser.add_argument("analysisDir", help='Cholla analysis output directory name', type=str)

    parser.add_argument("studyDir", help='Directory to place logs and plots', type=str)

    parser.add_argument("dlogk", help='Differential log of step-size for power spectrum k-modes', type=float)

    parser.add_argument("nOutputsStr", help='String of outputs delimited by comma', type=str)

    parser.add_argument('-v', '--verbose', help='Print info along the way', 
                        action='store_true')

    parser.add_argument('-n', '--name_study', help='Name of study', type=str)

    return parser


def check_dirsNfiles(skewers_DirPath, analysis_DirPath, study_DirPath, nOutputs, verbose=False):
    '''
    Take a look at the arguments and ensure that the required directories and 
        files are present
    
    Args:
        skewers_DirPath (Path): path to directory holding skewer files
        analysis_DirPath (Path): path to directory holding analysis files
        study_DirPath (Path): path to study directory
        nOutputs (arr): array of outputs for which to calculate optical depth for
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    success_str = "\t--- It lives ! :D ---"
    fail_str = "\t--- It doesn't exist ! D: ---"

    # First check directories
    if verbose:
        print(f"--- Making sure skewer directory exists : {skewers_DirPath} ---")
    if not skewers_DirPath.is_dir():
        if verbose:
            print(fail_str)
        return False
    elif verbose:
        print(success_str)

    if verbose:
        print(f"--- Making sure analysis directory exists : {analysis_DirPath} ---\n\n")
    if not analysis_DirPath.is_dir():
        if verbose:
            print(fail_str)
        return False
    elif verbose:
        print(success_str)

    if verbose:
        print(f"--- Making sure study directory exists : {study_DirPath} ---\n\n")
    if not study_DirPath.is_dir():
        if verbose:
            print(fail_str)
        return False
    elif verbose:
        print(success_str)

    # make sure file exists for each output
    for nOutput in nOutputs:
        skewer_fPath = skewers_DirPath / f"{nOutput:.0f}_skewers.h5"
        analysis_fPath = analysis_DirPath / f"{nOutput:.0f}_analysis.h5"

        if verbose:
            print(f"--- Making sure skewer file exists : {skewer_fPath} ---")

        if not skewer_fPath.is_file():
            if verbose:
                print(fail_str)
            return False
        elif verbose:
            print(success_str)
        
        if verbose:
            print(f"--- Making sure analysis file exists : {analysis_fPath} ---")
        if not analysis_fPath.is_file():
            if verbose:
                print(fail_str)
            return False
        elif verbose:
            print(success_str)

    if verbose:
        print(f"\n\n--- Congrats ! All required files and directories exist ---")
    
    return True


def create_optdepth_slurm(optdepth_pyscript_fPath, optdepth_slurm_fPath, skewers_DirPath, scriptlog_DirPath, studyName, nOutputs, verbose=False):
    '''
    Create optdepth.slurm file that will complete the optical depth calculations

    Args:
        optdepth_pyscript_fPath (Path): path to optical depth script
        optdepth_slurm_fPath (Path): path to place optical depth slurm file
        skewers_DirPath (Path): path to directory holding skewer files
        scriptlog_DirPath (Path): path to where logs for job will reside
        studyName (str): name to prepend job-name
        nOutputs (arr): array of outputs for which to calculate optical depth for
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    if verbose:
        print(f"--- Creating optical depth slurm file ---")

    # convert array to string, clip out brackets, remove whitespace
    array_str = np.array2string(nOutputs, separator=',')[1:-1]
    array_str = ''.join(array_str.split())

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-opticaldepth      # Job name
#SBATCH --partition=cpuq # Partition name
#SBATCH --account=cpuq   # Account name
#SBATCH --exclusive             # Exclusive use of the node
#SBATCH --ntasks=1              # Number of MPI ranks
#SBATCH --array={array_str}            # Array of job ids
#SBATCH --ntasks-per-node=1     # How many tasks on each node
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output={scriptlog_DirPath}/optdepth_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''

    bash_load = '''

module load slurm
module load python

module list

'''

    bash_scriptnfile = f'''
scriptPath="{optdepth_pyscript_fPath}"

skewerFile="{skewers_DirPath}/"$SLURM_ARRAY_TASK_ID"_skewers.h5"
'''

    bash_srun = f'''

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPath $skewerFile -l -v
'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_scriptnfile + bash_srun
    with optdepth_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have optical depth calculating slurm file : {optdepth_slurm_fPath} ---")

    return


def create_powspec_slurm(powspeccalc_pyscript_fPath, powspecplot_pyscript_fPath, powspec_slurm_fPath, dlogk, skewers_DirPath, analysis_DirPath, scriptlog_DirPath, plots_DirPath, studyName, nOutputs, verbose=False):
    '''
    Create powspec.slurm file that will complete the transmitted flux power spectrum calculations
        and plot each.

    Args:
        powspeccalc_pyscript_fPath (Path): path to power spectra calculation script
        powspecplot_pyscript_fPath (Path): path to power spectra plot script
        powspec_slurm_fPath (Path): path to place power spectra calculation slurm file
        dlogk (float): differential step in log k-space
        skewers_DirPath (Path): path to directory holding skewer files
        analysis_DirPath (Path): path to directory holding analysis files
        scriptlog_DirPath (Path): path to where logs for job will reside
        plots_DirPath (Path): path to where ALL power spectra difference plots will be saved
        studyName (str): name to prepend job-name
        nOutputs (arr): array of outputs for which to calculate optical depth for
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    if verbose:
        print(f"--- Creating plotting slurm file ---")

    # convert array to string, clip out brackets, remove whitespace
    array_str = np.array2string(nOutputs, separator=',')[1:-1]
    array_str = ''.join(array_str.split())

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-powspec      # Job name
#SBATCH --partition=cpuq                    # Partition name
#SBATCH --account=cpuq                      # Account name
#SBATCH --exclusive                         # Exclusive use of the node
#SBATCH --ntasks=1                          # Number of MPI ranks
#SBATCH --array={array_str}                 # Array of job ids
#SBATCH --ntasks-per-node=1                 # How many tasks on each node
#SBATCH --time=24:00:00                     # Time limit (hh:mm:ss)
#SBATCH --output={scriptlog_DirPath}/powspec_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''
    
    bash_load = '''

module load slurm
module load python

module list

'''

    bash_scriptnfiles = f'''
scriptCalcPath="{powspeccalc_pyscript_fPath}"
scriptPlotPath="{powspecplot_pyscript_fPath}"

skewerFile="{skewers_DirPath}/"$SLURM_ARRAY_TASK_ID"_skewers.h5"
analysisFile="{analysis_DirPath}/"$SLURM_ARRAY_TASK_ID"_analysis.h5"
'''

    bash_outdirsnfnames = f'''
plotsDir="{plots_DirPath}"

outDir=$plotsDir"/powspec_plots"
fname="PowerSpectra_"$SLURM_ARRAY_TASK_ID".png"

outDirDiff=$plotsDir"/powspecdiff_plots"
fnameDiff="PowerSpectraDiff_"$SLURM_ARRAY_TASK_ID".png"

outDirDiffLog=$plotsDir"/powspecdiff_log_plots"
fnameDiffLog="PowerSpectraLogDiff_"$SLURM_ARRAY_TASK_ID".png"

if [ ! -d "$outDir" ]; then
  echo "$outDir does not exist. Creating it"
  mkdir $outDir
fi

if [ ! -d "$outDirDiff" ]; then
  echo "$outDirDiff does not exist. Creating it"
  mkdir $outDirDiff
fi

if [ ! -d "$outDirDiffLog" ]; then
  echo "$outDirDiffLog does not exist. Creating it"
  mkdir $outDirDiffLog
fi
'''


    bash_srun = f'''
dlogk={dlogk}

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptCalcPath $skewerFile $dlogk -v -c

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPlotPath $skewerFile $analysisFile -v

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPlotPath $skewerFile $analysisFile -v -d

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPlotPath $skewerFile $analysisFile -v -d -l


'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_scriptnfiles + bash_srun
    with powspec_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have power spectra calc and plotting slurm file : {powspec_slurm_fPath} ---")
    
    return




def create_powspecplotcombo_slurm(powspecdiffcombo_pyscript_fPath, powspecdiffcombo_slurm_fPath, dlogk, skewers_DirPath, analysis_DirPath, scriptlog_DirPath, plots_DirPath, studyName, nOutputs, verbose=False):
    '''
    Create powspecdiff_combo.slurm file that will complete the optical depth calculations

    Args:
        powspecdiffcombo_pyscript_fPath (Path): path to power spectra difference plotting script
        powspecdiffcombo_slurm_fPath (Path): path to place power spectra difference combination slurm file
        dlogk (float): differential step in log k-space
        skewers_DirPath (Path): path to directory holding skewer files
        analysis_DirPath (Path): path to directory holding analysis files
        scriptlog_DirPath (Path): path to where logs for job will reside
        plots_DirPath (Path): path to where ALL power spectra difference plots will be saved
        studyName (str): name to prepend job-name
        nOutputs (arr): array of outputs for which to calculate optical depth for
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    # current implementation works only with 12 outputs to plot
    assert nOutputs.size == 12

    if verbose:
        print(f"--- Creating plotting combo slurm file ---")

    # convert array to string, clip out brackets
    array_str = np.array2string(nOutputs, separator=' ')[2:-1]

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-powspecdiff-combo      # Job name
#SBATCH --partition=cpuq # Partition name
#SBATCH --account=cpuq   # Account name
#SBATCH --exclusive             # Exclusive use of the node
#SBATCH --ntasks=1              # Number of MPI ranks
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output={scriptlog_DirPath}/powspecdiff_combo_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''
    
    bash_load = '''

module load slurm
module load python miniforge

module list

source activate /data/users/digarza/myconda_envs/cholla_analysis/
'''

    bash_script = f'''
scriptPath="{powspecdiffcombo_pyscript_fPath}"

skewersDir="{skewers_DirPath}"
analysisDir="{analysis_DirPath}"
'''


    bash_outdirsnfnames = f'''
plotsDir="{plots_DirPath}"

outDir=$plotsDir
fname="PowerSpectraDiff_ALL.png"
fnameLog="PowerSpectraLogDiff_ALL.png"
'''

    bash_srun = f'''
dlogk={dlogk}
nOutputStr="{array_str}"

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPath $skewersDir $analysisDir $dlogk $nOutputStr -v -o $outDir -f $fname

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPath $skewersDir $analysisDir $dlogk $nOutputStr -v -l -o $outDir -f $fnameLog

'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_script + bash_outdirsnfnames + bash_srun
    with powspecdiffcombo_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have power spectra difference plotting combined slurm file : {powspecdiffcombo_slurm_fPath} ---")

    return


def create_runstudy_slurm(runstudy_slurm_fPath, optdepth_slurm_fPath, powspecdiff_slurm_fPath, powspecdiffcombo_slurm_fPath, scriptlog_DirPath, studyName, verbose=False):
    '''
    Create run_study.slurm file that will submit all required jobs

    Args:
        runstudy_slurmPath (Path): path to place study running slurm file
        optdepth_slurm_fPath (Path): path to optical depth slurm file
        powspecdiff_slurm_fPath (Path): path to power spectra difference slurm file
        powspecdiffcombo_slurmPath (Path): path to power spectra difference combination slurm file
        scriptlog_DirPath (Path): path to where logs for job will reside
        studyName (str): name to prepend job-name
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    if verbose:
        print(f"--- Creating run study slurm file ---")

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-runstudy      # Job name
#SBATCH --partition=cpuq # Partition name
#SBATCH --account=cpuq   # Account name
#SBATCH --exclusive             # Exclusive use of the node
#SBATCH --ntasks=1              # Number of MPI ranks
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output={scriptlog_DirPath}/runstudy_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''

    bash_load = '''

module load slurm

module list

'''

    bash_sbatch = f'''
optDepthJob="{optdepth_slurm_fPath}"
powSpecDiffJob="{powspecdiff_slurm_fPath}"
powSpecDiffComboJob="{powspecdiffcombo_slurm_fPath}"

job_id=$(sbatch --parsable $optDepthJob)

echo "Submitted optical depth job "$job_id

job_id2=$(sbatch --parsable -d afterok:$job_id $powSpecDiffJob)

echo "Submitted power spectra job "$job_id2

job_id3=$(sbatch --parsable -d afterok:$job_id $powSpecDiffComboJob)

echo "Submitted power spectra combo job "$job_id3
'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_sbatch
    with runstudy_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have slurm file to run entire study : {runstudy_slurm_fPath} ---")

    return


def create_clearskewers_slurm(clearskew_pyscript_fPath, clearskewers_slurm_fPath, skewers_DirPath, clearlog_DirPath, studyName, nOutputs, verbose=False):
    '''
    Create optdepth.slurm file that will complete the optical depth calculations

    Args:
        clearskew_pyscript_fPath (Path): path to skewer clearing script
        clearskewers_slurm_fPath (Path): path to place clearing skewer slurm file
        skewers_DirPath (Path): path to the skewers directory
        clearlog_DirPath (Path): path to where logs for job will reside
        studyName (str): name to prepend job-name
        nOutputs (arr): array of outputs for which to calculate optical depth for
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    if verbose:
        print(f"--- Creating clear skewer slurm file ---")

    # convert array to string, clip out brackets, remove whitespace
    array_str = np.array2string(nOutputs, separator=',')[1:-1]
    array_str = ''.join(array_str.split())

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-clearskew      # Job name
#SBATCH --partition=cpuq # Partition name
#SBATCH --account=cpuq   # Account name
#SBATCH --exclusive             # Exclusive use of the node
#SBATCH --ntasks=1              # Number of MPI ranks
#SBATCH --array={array_str}            # Array of job ids
#SBATCH --ntasks-per-node=1     # How many tasks on each node
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output={clearlog_DirPath}/clearskew_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''

    bash_load = '''

module load slurm
module load python miniforge

module list
'''

    bash_scriptnfile = f'''
scriptPath="{clearskew_pyscript_fPath}"

skewerFile="{skewers_DirPath}/"$SLURM_ARRAY_TASK_ID"_skewers.h5"
'''

    bash_srun = f'''

srun -N 1 -n 1 -c 1 --cpu-bind=cores --exclusive --partition=cpuq --account=cpuq python3 $scriptPath $skewerFile -v
'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_scriptnfile + bash_srun
    with clearskewers_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have skewer cleaning slurm file : {clearskewers_slurm_fPath} ---")

    return


def create_clearstudy_slurm(clearstudy_slurm_fPath, clearskewers_slurm_fPath, scriptlog_DirPath, plots_DirPath, clearlog_DirPath, studyName, verbose=False):
    '''
    Create clear_study.slurm file that will remove all optical depth calcualtion 
        datasets, plots directory, and log directory. Logs to complete this
        are saved in a seperate clearinglogs directory
        
        CAUTION: there's calls to rm -rf written (!)

    Args:
        clearstudy_slurm_fPath (Path): path to place clearing study slurm file
        clearskewers_slurm_fPath (Path): path to clearing skewers slurm file
        scriptlog_DirPath (Path): path to where logs for study were residing
        plots_DirPath (Path): path to where ALL power spectra difference plots are saved
        clearlog_DirPath (Path): path to where logs for job will reside
        studyName (str): name to prepend job-name
        verbose (bool): (optional) whether to print important information
    Returns:
        ...
    '''
    if verbose:
        print(f"--- Creating clear study slurm file ---")

    bash_header = f'''#!/bin/bash
#SBATCH --job-name={studyName}-clearstudy      # Job name
#SBATCH --partition=cpuq # Partition name
#SBATCH --account=cpuq   # Account name
#SBATCH --exclusive             # Exclusive use of the node
#SBATCH --ntasks=1              # Number of MPI ranks
#SBATCH --time=24:00:00         # Time limit (hh:mm:ss)
#SBATCH --output={clearlog_DirPath}/clearstudy_%j.log     # Standard output and error log
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=digarza@ucsc.edu

'''

    bash_load = '''

module load slurm

module list

'''
    bash_rm = f'''
plotsDir="{plots_DirPath}"
logsDir="{scriptlog_DirPath}"

echo "Removing plots directory files"
rm -rf $plotsDir/*

echo "Removing study logs directory files"
rm -rf $logsDir/*.log
'''

    bash_sbatchskew = f'''
clearSkewJob="{clearskewers_slurm_fPath}"
echo "Calling in clear skewers job"
sbatch $clearSkewJob

echo "Study info cleared !"
'''

    # combine text and write to file
    bash_txt = bash_header + bash_load + bash_rm + bash_sbatchskew
    with clearstudy_slurm_fPath.open('w') as slurmfile:
        slurmfile.write(bash_txt)

    if verbose:
        print(f"\t --- Woo hoo! We now have study clearing slurm file : {clearstudy_slurm_fPath} ---")

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

    # ensure dlogk is reasonable
    assert args.dlogk > 0

    # convert relative path to skewer file name to absolute file path
    skewers_DirPath = Path(args.skewDir)
    analysis_DirPath = Path(args.analysisDir)
    study_DirPath = Path(args.studyDir)
    skewers_DirPath = skewers_DirPath.resolve()
    analysis_DirPath = analysis_DirPath.resolve()
    study_DirPath = study_DirPath.resolve()

    # make sure we have some name. can I allocate that memory myself? like if name_study is undefined, can I name it?
    if args.name_study:
        studyName = args.name_study
    else:
        studyName = args.studyDir.split('/')[-1] 
        if args.verbose:
            print(f"--- Name for study was not specified, using study directory : {studyName} ---")

    # parse nOutputsStr, get list of outputs, convert to np array
    nOutputsStr_lst = args.nOutputsStr.split(',')
    nOutputs = np.zeros(len(nOutputsStr_lst), dtype=np.int64)
    for n, nOutputStr in enumerate(nOutputsStr_lst):
        # cast into int
        nOutputs[n] = int(nOutputStr)

    # make sure all files and directories input exist
    assert check_dirsNfiles(skewers_DirPath, analysis_DirPath, study_DirPath, nOutputs, args.verbose)

    # define scripts we are going to be using in slurm files, assume they're in same directory
    curr_pyscript_fPath = Path(__file__).parent.resolve()
    optdepth_pyscript_fPath = curr_pyscript_fPath / "optdepth.py"
    powspecdiff_pyscript_fPath = curr_pyscript_fPath / "powspec_diff.py"
    powspec_combo_pyscript_fPath = curr_pyscript_fPath / "powspecdiff_all.py"
    clearskew_pyscript_fPath = curr_pyscript_fPath / "clear_skew.py"

    # make sure all python scripts exist
    for pyscript_fPath in [optdepth_pyscript_fPath, powspecdiff_pyscript_fPath, powspec_combo_pyscript_fPath, clearskew_pyscript_fPath]:
        print(f"--- Making sure python script exists : {pyscript_fPath}---")
        assert pyscript_fPath.is_file()

    # create script logs directory
    scriptlog_DirPath = study_DirPath / "scriptlogs"
    if scriptlog_DirPath.is_dir() and args.verbose:
        print(f"--- Directory holding log files already created - Please remove : {scriptlog_DirPath} ---")
    scriptlog_DirPath.mkdir(parents=False, exist_ok=False)
    
    # create clearing logs directory
    clearlog_DirPath = study_DirPath / "clearinglogs"
    if not clearlog_DirPath.is_dir():
        if args.verbose:
            print(f"--- Creating directory holding clearing log files---")
        clearlog_DirPath.mkdir(parents=False, exist_ok=False)
    else:
        print(f"--- Directory holding clearing log files already created ---")

    # create plots directory
    plots_DirPath = study_DirPath / "PowerSpectraPlotsDiff"
    if plots_DirPath.is_dir() and args.verbose:
        print(f"--- Directory holding plots of power spectra differences already created - Please remove : {plots_DirPath} ---")
    plots_DirPath.mkdir(parents=False, exist_ok=False)

    # create the optical depth slurm file
    optdepth_slurm_fPath = study_DirPath / "optdepth.slurm"
    create_optdepth_slurm(optdepth_pyscript_fPath, optdepth_slurm_fPath, skewers_DirPath, scriptlog_DirPath, studyName, nOutputs, args.verbose)

    # create the power spectra slurm file
    powspecdiff_slurm_fPath = study_DirPath / "powspec_diff.slurm"
    create_powspecplot_slurm(powspecdiff_pyscript_fPath, powspecdiff_slurm_fPath, args.dlogk, skewers_DirPath, analysis_DirPath, scriptlog_DirPath, plots_DirPath, studyName, nOutputs, args.verbose)

    # create the power spectra combination slurm file
    powspecdiff_combo_slurm_fPath = study_DirPath / "powspec_diff_combo.slurm"
    # define outputs to include in combo & make sure they're subset of input
    nOutputsCombo = np.array([8,9,10,11,12,13,14,15,16,17,18,19], dtype=np.int64)
    assert set(nOutputsCombo).issubset(nOutputs)
    create_powspecplotcombo_slurm(powspec_combo_pyscript_fPath, powspecdiff_combo_slurm_fPath, args.dlogk, skewers_DirPath, analysis_DirPath, scriptlog_DirPath, plots_DirPath, studyName, nOutputsCombo, args.verbose)

    # create final run all scripts slurm file
    runstudy_slurm_fPath = study_DirPath / "run_study.slurm"
    create_runstudy_slurm(runstudy_slurm_fPath, optdepth_slurm_fPath, powspecdiff_slurm_fPath, powspecdiff_combo_slurm_fPath, scriptlog_DirPath, studyName, args.verbose)

    # create slurm file to clear out all optical depth datasets added to skewers
    clearskewers_slurm_fPath = study_DirPath / "clear_skewers.slurm"
    create_clearskewers_slurm(clearskew_pyscript_fPath, clearskewers_slurm_fPath, skewers_DirPath, clearlog_DirPath, studyName, nOutputs, args.verbose)

    # create slurm file to clear out all aspects from study
    clearstudy_slurm_fPath = study_DirPath / "clear_study.slurm"
    create_clearstudy_slurm(clearstudy_slurm_fPath, clearskewers_slurm_fPath, scriptlog_DirPath, plots_DirPath, clearlog_DirPath, studyName, args.verbose)


if __name__=="__main__":
    main()

