# Installing the pipeline

## Linux

For now, the pipeline was exclusively tested on Linux. There are not previous requirements before installing.

1. Clone the pipeline repo from github :

```bash
cd
git clone https://github.com/spsalmon/towbintools_pipeline.git
```

2. Install micromamba : BE CAREFUL, THE PIPELINE EXPECTS YOU TO CHOSE THE BASE OPTIONS AND YES ALL THE TIME. Then, restart your shell. You can skip this part if you already have it installed.

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
```

3. Run the installation script :

```bash
cd ~/towbintools_pipeline
bash install_pipeline.sh
```

This will create a micromamba environment called towbintools, containing all required packages. Now you're all set, click here to learn how to [run your first pipeline](https://spsalmon.github.io/towbintools_pipeline/runningfirstpipeline/). Click here to learn how to [update the pipeline](https://spsalmon.github.io/towbintools_pipeline/update/)

## Windows

1. You may have to install git first, follow instructions given [here](https://git-scm.com/install/windows). For all the other steps, please run them using Git Bash (not CMD, not Powershell).

2. Using Git Bash, clone the pipeline repo from github :

```bash
cd
git clone https://github.com/spsalmon/towbintools_pipeline.git
```

3. Install micromamba, using Git Bash: BE CAREFUL, THE PIPELINE EXPECTS YOU TO CHOSE THE BASE OPTIONS AND YES ALL THE TIME. You can also ollow instructions given [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Then restart Git Bash manually.

4. Using Git Bash, run the installation script

```bash
cd ~/towbintools_pipeline
bash install_pipeline.sh
```

Another option is to use Windows Subsystem for Linux (WSL, follow [instructions to install](https://learn.microsoft.com/en-us/windows/wsl/install)) and follow the Linux instructions inside of WSL.
