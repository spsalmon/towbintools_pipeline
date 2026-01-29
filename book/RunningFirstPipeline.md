# Running your first pipeline

The pipeline is configure using YAML files. After installing the pipeline you will find an example of a working pipeline for measuring organ and body size over development. Let's break it down!
For now, the basic assumption is that your images follow this specific naming scheme : TimeX_PointY_... where Time refers to the index in your time loop and Point is the number that we will use as
the unique identifier for the position (individual worm if each position contains one worm). All images (planes, channels, etc.) for a given position at a given time should be stored in the same OME-TIFF file.

'''yaml
experiment_dir: "/mnt/towbin.data/shared/spsalmon/pipeline_test_folder/"
analysis_dir_name: "analysis"
raw_dir_name: "raw"
report_format: "parquet"
'''

- **experiment_dir** : the root of your experiment
- **analysis_dir_name** : the name of the directory where all the analysis files (segmentation masks, quantifications, etc.) will be saved. You can change it to run multiple different analysis of the same experiment
- **raw_dir_name** : directory where all your raw images are saved
- **report_format**: either "csv" or "parquet". Parquet files will be much smaller than CSVs (usefull for big experiments), but are less convenient to edit.
