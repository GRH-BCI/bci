# CCA

Library and helper scripts for SSVEP classification using various forms of Canonical Correlation Analysis.

## Installation

### GitHub SSH keys

For faster CCA coefficient calculations I use a fork of `svcca`, which is a Google library built for analyzing deep neural networks.
To eventually be able to install it, you have to make an ssh key and add it to your GitHub account (see https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account):

    ssh-keygen -t ed25519 -C "git@github.com"
    start-ssh-agent
    clip < ----.pub

then paste the clipboard contents in the "Key" field after clicking on "New SSH Key" in the "SSH and GPG keys" section of your GitHub  settings.

You can alternatively add it as a deploy key to the repository directly at https://github.com/GRH-BCI/svcca/settings/keys.
Note that a single ssh key cannot be used for both at once.

### Python environment

Next, create a Python environment with Anaconda (which we unfortunately have to use here because we need `faiss-gpu` for fast clustering in the CACC algorithm).
Download Anaconda, then, from the Anaconda prompt:

    start-ssh-agent
    ssh git@github.com
    conda env create -f environment.yml


### PostgreSQL

The library used for parameter searching, optuna, requires postgresql (it can technically run on other backends like sqlite, but in practice sqlite can't handle the amount of concurrency and large file sizes that my parameter searches can involve).
I use the default database `postgres` running on `localhost:5432` with the password `i5gMr!Pfcdm$dn8YqhTf#$hL?jkb` (obviously this is a terrible way to use postgresql, but it's easy to set up and I don't actually know how to use it any other way).
To set this up, download postgres from https://www.postgresql.org/download/ and enter `i5gMr!Pfcdm$dn8YqhTf#$hL?jkb` for the superuser password.

### Dataset

The various scripts expect there to be a folder named `C:\datasets\wearable-sensing` that contains subfolders for each trial.
Download the data from Google Drive and put it in this folder.

## Code overview

### bci/bci/

The `bci` directory contains helper scripts for SSVEP classification applications.
I give a brief overview of its various components in the following sections.

 * **dsi_input.pyd**: this file, together with the accompanying dlls, is a dynamic Python library that interfaces with the DSI-24 headset. See [DSIInputPy](https://github.com/GRH-BCI/DSIInputPy) for more details.
 * **eeg.py**: this file contains the dataclass `EEG`, which stores EEG data and associated metadata. It handles loading EEG from a collection of `.csv` files, as well as some simple filtering.
 * **gui.py**: contains `BCIGUI`, a class that provides a simple interface for SSVEP training built with `pygame`.
 * **leds.py**: contains `LEDs`, a class which manages turning strobe LEDs on or off by communicating with the attached Arduino.
 * **util.py**: contains various miscellaneous utilities, including `InputDistributor` which takes EEG input from a `DSIInput` instance and passes it around to various `InputListener` instances, which can write the EEG data to a file (`FileRecorder`) or pass it to a predictive model (`RealtimeModel`).

### bci/fbcca/

The `fbcca` directory contains scripts pertaining to Filter Bank Canonical Correlation Analysis.
It implements the Chen et al.'s [Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface](https://doi.org/10.1088/1741-2560/12/4/046008).

The FBCCA algorithm is implemented in `fbcca/cca.py`. `fbcca/param_search.py` 
