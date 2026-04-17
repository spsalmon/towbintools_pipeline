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

The easiest way to get things to work on Windows is to use Windows Subsystem for Linux (WSL) : follow [instructions to install](https://learn.microsoft.com/en-us/windows/wsl/install)

1. Inside of a Powershell terminal run as administrator, run :

```PowerShell
wsl --install -d Ubuntu
wsl --set-default-version 2
```
2. To use WSL, inside of a terminal, run :

```PowerShell
wsl
```

3. Follow Linux instructions inside of WSL.
