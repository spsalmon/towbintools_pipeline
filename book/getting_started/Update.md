# Updating the pipeline

## Fast explanation

For convenience, I created a script that will do all operations explained in the second section. To update the pipeline:

1. Deactivate the environment and close scripts, notebooks, terminals, where it's activated
2. Run the update script

```bash
cd ~/towbintools_pipeline #(or where you installed the pipeline)
bash update_pipeline.sh
```

Optionally, you can run

```bash
bash update_pipeline.sh --pipeline-only
```

To update only the pipeline scripts and not the underlying packages

## Longer explanation

Updating the pipeline is done in two parts. First, update the repository using the source on github. This can be done different ways:
1. using git pull
2. by resetting the repository based on the upstream content, using

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

If what you want is to reset the repository to how it is on github, use option 2.

Then, you need to update the packages. For consistency's sake, the pipeline uses conda lock files. They ensure that the package versions you end up having is exactly the same as what was tested during development. To update your environment according to a lock file, run

```bash
cd ~/towbintools_pipeline
micromamba run -n towbintools conda-lock install --name towbintools ./requirements/conda-lock.yml
```

## Troubleshooting

If for some weird reason, you end up with a broken environment, you can always delete it and create it again using the lock file. To do so, run:

```bash
micromamba env remove -n towbintools
cd ~/towbintools_pipeline
bash install_pipeline.sh
```

You may also want to update the towbintools package itself manually. To do so, run:

```bash
micromamba run -n towbintools pip install -U towbintools
```
