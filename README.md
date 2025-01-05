# cholla_lya_scripts

Python scripts to study the Lyman-alpha Forest in cosmological Cholla simulations

We would like to study the optical depth calculation and transmitted flux power spectrum for a cosmological Cholla simulation, using different optical depth criteria. Our end goal will be showing the difference between a calculated transmitted flux power spectrum against the one outputted from the analysis file. 

What exactly are we comparing? Well ... [include explanation from overleaf] 

We will have a couple outputs:

1. ``scriptlogs/`` - directory holding information from optical depth calculation
2. ``PowerSpectraPlotsDiff/`` - directory holding 2 plots and directories of individual plots, displaying the relative error of the transmitted flux power spectrum, with and without log-space

We need 5 inputs:

1. ``skewers/`` - directory holding skewer outputs
2. ``analysis/`` - directory holding analysis outputs
3. ``study/`` - directory holding outputs
4. ``dlogk`` - differential step in log k-space
5. ``OutputStr`` - string of outputs to study, delimited by commas


How do we run the speed-up study? First we create all of the directories and slurm files required using `setup_study.py`

```
$ python3 setup_study.py $skewersDir $analysisDir $studyDir $dlogk $OutputStr -v
```

where

``-v`` flags the script to be verbose throughout the calculation
``-n`` flags the name of the study to be something other than the last directory of ``$studyDir``


After running this script we then have

```bash
/study/
├── run_study.slurm
├── optdepth.slurm
├── powspec_diff.slurm
├── powspec_diff_combo.slurm
├── clear_skewers.slurm
├── clear_study.slurm
├── PowerSpectraPlotsDir
│   └── ...
└── scriptlogs
│   └── ...
└── clearinglogs
│   └── ...
```

woah! What has been placed in this study directory? Well we have 6 slurm files:

1. ``optdepth.slurm`` - calls optical depth calculation python script to add optical depths to skewer files
2. ``powspec_diff.slurm`` - calls power spectra difference python script to plot the relative error of transmitted flux power spectrum
3. ``powspecdiff_combo.slurm`` - calls power spectra difference combopython script to plot the relative error of transmitted flux power spectrum from different snapshot outputs
4. ``run_study.slurm`` - submits previous 3 slurm files with appropriate dependencies
5. ``clear_skewers.slurm`` - calls clear skewers python script to remove optical depths from skewer files, reset back to regular Cholla outputs
6. ``clear_study.slurm`` - calls slurm script 5 and clears out the plots and script directories

Well what will be in the three directories?

1. ``scriptlogs`` - saves the outputs of how the optical depth was calculated and statistics of relative and absolute error
2. ``PowerSpectraPlotsDir`` - saves individual and combined transmitted flux power spectra relative error without and with log-space
3. ``clearinglogs`` - saves the outputs from clearing slurm files






