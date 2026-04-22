# Gathering a Segmentation Dataset

Like many other parts of the pipeline, this is also configured through a YAML configuration file. The goal of this script is to aggregate a balanced and representative dataset of images. The configuration file allows you to specify various parameters that control how the dataset is gathered, including where to look for the data, what types of images to include, and how to balance the dataset across different stages of development.

This script expects (as the rest of the pipeline) the images to be stored in a single "raw" directory for each experiment. Some information should be contained in the name of the experiment directory :
- the name of the microscope used
- the magnification used
- the name(s) of the strain(s) (if applicable)
- the date of the experiment

Example : '20251008_ZIVA_40x_470_471_447_449_25C', where all raw images are stored in '20251008_ZIVA_40x_470_471_447_449_25C/raw'

## Options

- storage_path : The base path where the data is stored. This is the root directory that contains all the subdirectories for different experiments and datasets.
- valid_subdirectories : A list of subdirectories within the storage path that should be considered when gathering the dataset. Only data from these subdirectories will be included in the dataset.
- database_path : Output path where the gathered dataset will be stored. This is where the final dataset will be saved after processing.

Example :
```yaml
storage_path: "/mnt/towbin.data/shared"
valid_subdirectories: ["spsalmon", "kstojanovski", "igheor"]
database_path: "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/datasets/chamber_segmentation"
```

- database_configs : This section allows you to specify configurations for different types of images (e.g. brightfield, fluorescence, etc.) or structures (e.g. body, pharynx, etc.). For each type of image, you can specify parameters such as the size of the dataset, the proportions of different stages of development, the channels to include, and the strains and magnifications to consider.

- stage_proportions : This parameter allows you to specify the proportions of different stages of development (e.g. L1, L2, L3, L4, adult) in the dataset. If set to null, the script will gather images randomly without considering the stage of development. If specified, the script will try to gather images in a way that matches the specified proportions as closely as possible. For this to be possible, ecdysis times should be available in the analysis filemaps of your experiments. This parameter can either be in the form {'eggs': 0.1, 'L1': 0.2, 'L2': 0.2, 'L3': 0.2, 'L4': 0.2, 'adult': 0.1}, in this information on hatch and all molts is required or in the form {'larvae': 0.5, 'adult': 0.5} where only M4 information is needed.

Example :
```yaml
database_configs: {
  'brightfield': {
      'size': 1000,
      'stage_proportions': {'eggs': 0.1, 'L1': 0.2, 'L2': 0.2, 'L3': 0.2, 'L4': 0.2, 'adult': 0.1},
      'scope_proportions': {'ziva': 1.0},
      'channel': [0],
      'strains': null,
      'magnifications': ["40x", "60x"],
  },
}
```

- extra_adulthood_time : In most microchamber experiments, adult worms will start to lay eggs and have progeny, making it difficult to accurately segment the adult worms, so this data is generally ignored. This parameter allows you to specify the number of time points after M4 to consider for inclusion into the dataset.
- n_picks_per_stage : If stage_proportions is specified, this parameter allows you to specify the number of images to gather for each stage of development. This can be useful to ensure that you have a sufficient number of images for each stage, especially if some stages are less represented in your experiments.

Example :
```yaml
extra_adulthood_time: 40
n_picks_per_stage: 50
```

- valid_scopes : This parameter allows you to specify which scopes to consider when gathering the dataset. Only data from these scopes will be included in all datasets.
- scopes_alt_names : This parameter allows you to specify alternative names for the microscopes. This is useful if the names of the scopes in your experiment directories do not exactly match the valid scope names specified in the previous parameter.

Example :
```yaml
valid_scopes: ["ziva"]
scopes_alt_names: {
  'crest': ['Crest', 'crest', 'CREST'],
  'squid': ['Squid', 'squid', 'SQUID'],
  'ti2': ['Ti2', 'ti2', 'TI2', 'orca', 'Orca', 'ORCA'],
  'ziva': ['ziva', 'ZIVA', 'Ziva']
}
```

- keyword_to_exclude : This parameter allows you to specify keywords that, if present in the experiment directory name, will lead to the exclusion of that experiment from the dataset.
- keyword_to_include : This parameter allows you to specify keywords that, if present in the experiment directory name, will lead to the automatic inclusion of that experiment in the dataset, even if it would otherwise be excluded based on other criteria.
- experiments_to_consider : This parameter allows you to specify a list of specific experiment directories to consider when gathering the dataset. Experiments not in this list will never be included in the dataset.
- experiments_to_always_include : This parameter allows you to specify a list of specific experiment directories that should always be included in the dataset, regardless of other criteria.
- experiments_to_exclude : This parameter allows you to specify a list of specific experiment directories that should always be excluded from the dataset, regardless of other criteria.

Example:
```yaml
keywords_to_exclude: ["exclude", "fail", "failure", "crash"]
keywords_to_include: ["lifespan"]

experiments_to_consider: null

experiments_to_always_include: []
experiments_to_exclude: [
  "20241111_squid_10x_wBT318_NaCl",
  "20241707_souvik_w318_ev_mex3_squid",
  "20240515_squid1_10x_wbt160_25C_2024-05-15_15-31-45.506118",
  "20240212_squid_wbt318_Nacl",
  "20241912_squid_10x_wBT_344_415",
  "20252401_squid_10x_wBT318_reproduction",
  "20252001_squid_10x_wBT318_NaCl",
]
```
