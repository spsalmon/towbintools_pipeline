# Installing the pipeline

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

This will create a micromamba environment called towbintools, containing all required packages. Now you're all set, click here to learn how to [run your first pipeline](link).
