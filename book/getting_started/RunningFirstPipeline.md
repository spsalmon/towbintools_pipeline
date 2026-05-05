# Running your first pipeline

## Building your configuration

The pipeline is configured using YAML files.
For now, the basic assumption is that your images follow this specific naming scheme : TimeX_PointY_(...).tiff where Time refers to the index in your time loop and Point is the number that we will use as
the unique identifier for the position (individual worm if each position contains one worm). All images (planes, channels, etc.) for a given position at a given time should be stored in the same OME-TIFF file. For more details on input formats and assumptions, head to [pipeline input](https://spsalmon.github.io/towbintools_pipeline/usage/pipelineinput/).

The pipeline works using atomic [building blocks](https://spsalmon.github.io/towbintools_pipeline/building-blocks/buildingblock/), head to the **Building Block** section to learn more about them.

After installing the pipeline you will find an example of a working configuration for measuring organ and body size over  in the 'configs/' directory. The default configuration in 'configs/config.yaml' will be reset to its upstream version every time you update.

Let's break it down!

```yaml
experiment_dir: "/mnt/towbin.data/shared/spsalmon/pipeline_test_folder/"
analysis_dir_name: "analysis"
raw_dir_name: "raw"
report_format: "parquet"
pixelsize: [ 0.65 ]
get_experiment_time: True
time_regex: 'Time(\d+)'
point_regex: 'Point(\d+)'
```

- **experiment_dir** : the root of your experiment
- **analysis_dir_name** : the name of the directory where all the analysis files (segmentation masks, quantifications, etc.) will be saved. You can change it to run multiple different analysis of the same experiment
- **raw_dir_name** : directory where all your raw images are saved
- **report_format** : either "csv" or "parquet". Parquet files will be much smaller than CSVs (usefull for big experiments), but are less convenient to edit
- **pixelsize** : physical size in µm of a pixel (depends on your microscope, camera and objective)
- **get_experiment_time** : if True, the actual time the images were acquired at will be extracted for the metadata. This can take quite a while but is very usefull for downstream analysis
- **time_regex** : the regular expression used to extract the time index from the filename. The default one works for filenames like TimeX_PointY_(...).tiff, but you can change it to fit your naming scheme. The part in brackets will be extracted as the time index.
- **point_regex** : same as time_regex but for the point index (unique identifier for each worm). The part in brackets will be extracted as the point index.

If you have different imaging modalities during your timelapse (let's say you acquire a picture of each worm every 10 minutes but also take a Z-stack every hour), you should split them in different raw directories (e.g. raw and raw_stack). You can then run a pipeline for each of those directories and merge them at the end simply by joining the two dataframes.

```yaml
use_slurm: True
sbatch_memory: 64G
sbatch_time: 0-48:00:00
sbatch_cpus: 32
sbatch_gpus: "rtx6000:1"
```

Those options control the amount of RAM, CPU cores, and GPU allocated to each building block, as well as their time limit. GPUs will only get allocated to jobs that can make use of them (segmentation, molt detection, or
custom script requiring GPU). If you don't have access to a cluster and want to run the pipeline on your local machine, simply set **use_slurm** to False and the building blocks will be run sequentially (NOT IMPLEMENTED YET).

```yaml
building_blocks:
  - "segmentation"
  - "segmentation"
  - "straightening"
  - "straightening"
  - "straightening"
  - "straightening"
  - "morphology_computation"
  - "morphology_computation"
  - "fluorescence_quantification"
  - "quality_control"
  - "quality_control"
  - "molt_detection"

rerun_segmentation: [ False ]
rerun_straightening: [ False ]
rerun_morphology_computation: [ False ]
rerun_quality_control: [ False ]
rerun_fluorescence_quantification: [ False ]
rerun_quality_control: [ False ]
rerun_molt_detection: [ False ]
```

- **building_blocks** : the list of atomic tasks that you want the pipeline to perform. In this case, 2 segmentations, 3 straightenings, etc.
- **rerun** : if False, images that were already processed will be skipped, only missing ones will be processed. For blocks like morphology_computation, the whole block is skipped if the resulting file "(...).csv" already exists. If true, everything is reprocessed.

After that, you should fill add the parameters for all of your building blocks. Those are described in detail [here](https://spsalmon.github.io/towbintools_pipeline/building-blocks/buildingblock/). Here is an example of configuration for the two segmentation blocks:

```yaml
# segmentation parameters
segmentation_column: [ 'raw' ]
segmentation_name_suffix: [ null ]
segmentation_method: [ "deep_learning" ]
segmentation_channels: [ [ 1 ], [ 0 ] ]


# deep learning segmentation parameters
model_path: [ "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/body/towbintools_medium/best_light.ckpt", "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/paper/pharynx/towbintools_medium/best_light.ckpt" ]
batch_size: [ 4 ]
```
## Running the pipeline

An example configuration will always be available in the 'configs/' directory of your pipeline installation. You can use it as a template to build your own configuration. This configuration file will automatically be updated to the latest version when you update the pipeline. You should therefore not use it as your working configuration, but rather copy it and give it a different name. You can then modify it to fit your needs. This copied version will not be overwritten when we update the original configuration file.

Once your configuration is finished, you can save it anywhere. Let's assume you saved it in ~/towbintools_pipeline/configs/my_configuration.yaml. I recommend picking a folder where you will centralize all your configuration. The configuration you run on an experiment will always be backed up in the analysis/report folder of said experiment. To run this specific configuration :

```bash
cd ~/towbintools_pipeline # or wherever you chose to put the pipeline folder
bash run_pipeline.sh -c configs/my_configuration.yaml
```

The -c argument is used to specify the path to the configuration to be run.

That's it ! Once you're happy with your config, all you need to do to fully analyze a new experiment is to change the experiment path in the config file and run this command.
