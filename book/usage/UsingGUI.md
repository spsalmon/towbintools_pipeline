# Using the GUI

To access the GUI, first modify the **FILEMAP_PATH** variable in `launch_gui.sh` to point to the filemap you want to open. Then run the following commands in your terminal:

```bash
bash launch_gui.sh
```

When opening a filemap in the GUI, we automatically create a backup in the same folder. The GUI then saves the annotated version of the filemap with an "_annotated" suffix. By default, the GUI always tries to load the annotated version if it exists. You can disable this behavior by switching **OPEN_ANNOTATED** to 0 in `launch_gui.sh`. If set to 0, it will always open exactly the filemap path provided. You may chose to have the GUI recompute all values at molt on launch by setting **RECOMPUTE_VALUES_AT_MOLT** to 1.

The Shiny GUI is a way to verify, annotate and explore your data in a more interactive way. It is meant to be used after all the parts of your pipeline finished running. The plot shows the measurement across time of the selected individual. You can select the feature you want to plot on the y axis using **Select column to plot**. Data points corresponding to errors as detected through the automated quality control appear as triangles. Eggs appear as squares and the rest as circles. You can click on a data point to jump to the corresponding time and image.You can select which image channel is being displayed and overlay segmentation masks using the dropdown menues on the right.

The GUI lets you see the molts detected by the automatic molt detection algorithm. Clicking one of the ecdysis buttons will annotate the current time as this molt exit. Doing so automatically interpolates the values of each series at the time of the molt (shown by a cross of the corresponding color). If you are unhappy with this interpolation, clicking **Select value at molt** will peg it to the values at the current time.

You can annotate special events using custom columns. If you type the name of a custom column in the text input, the next time you click **Annotate** will add it to the dataframe. You can then select which column you are annotating using the dropdown menu.

If for some reason you already have annotations for this experiment in another analysis filemap, you can import them to the current one using the **Import annotations** button. This will also recalculate all values at ecdysis.
