# Installing the pipeline

## Linux

For now, the pipeline was exclusively tested on Linux. There are not previous requirements before installing.

1. Clone the pipeline repo from github :

```bash
git clone https://github.com/spsalmon/towbintools_pipeline.git ~
```

2. Install micromamba and restart your shell. You can skip this part if you already have it installed.

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

1. You may have to install git first, follow instructions given [here](https://git-scm.com/install/windows)

2. Install micromamba, using Git Bash, or follow instructions given [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
