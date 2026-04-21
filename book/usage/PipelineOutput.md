# Pipeline Output

The pipeline outputs its results in different formats. Any block that produces images (e.g. segmentation masks, straightened images, etc.) will save them as OME-TIFF files in a subdirectory inside the analysis directory. Any block that produces quantifications (e.g. morphology_computation, fluorescence_quantification, etc.) will save them in a single report file in the analysis directory. If a a block would produce many quantifications per image (e.g. a value at every plane of a Z-stack), it will save them as one individual report file (per image) in a subdirectory inside the analysis directory.

Report files can be in either CSV or Parquet format, depending on the "report_format" parameter in the configuration file. Parquet files are much smaller than CSVs (usefull for big experiments), but are less convenient to edit (in Microsoft Excel for example).
