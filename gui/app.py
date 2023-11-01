from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
from towbintools.foundation import image_handling
from towbintools.foundation.detect_molts import compute_volume_at_time
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

import plotly.express as px
import plotly.graph_objs as go
import numpy as np

from time import perf_counter

filemap_path = '/mnt/external.data/TowbinLab/kstojanovski/20220401_Ti2_20x_160-182-190_pumping_25C_20220401_173300_429/analysis/report/analysis_filemap.csv'
filemap = pd.read_csv(filemap_path)

times = filemap['Time'].unique().tolist()
points = filemap['Point'].unique().tolist()

channels = image_handling.read_tiff_file(filemap['raw'].iloc[0]).shape[0]
list_channels = [f'Channel {i+1}' for i in range(channels)]

usual_columns = ['Time', 'Point', 'raw', 'analysis/ch2_seg', 'analysis/ch2_seg_str','Volume', 'WormType', 'HatchTime', 'VolumeAtHatch', 'M1', 'VolumeAtM1','M2', 'VolumeAtM2', 'M3', 'VolumeAtM3', 'M4', 'VolumeAtM4']
list_custom_columns = [column for column in filemap.columns.tolist() if column not in usual_columns]

molt_annotator = ui.column(7,
                  ui.row(
                      output_widget('volume_plot')),
                  ui.row(
                      ui.column(2, ui.row(ui.output_text('hatch_text')), ui.row(ui.input_action_button(
                          "set_hatch", "hatch")), ui.row(ui.input_action_button("reset_hatch", "Reset Hatch"))),
                      ui.column(2, ui.row(ui.output_text("m1_text")),
                                ui.row(ui.input_action_button("set_m1", "M1")),
                                ui.row(ui.input_action_button("reset_m1", "Reset M1"))),
                      ui.column(2, ui.row(ui.output_text("m2_text")),
                                ui.row(ui.input_action_button("set_m2", "M2")),
                                ui.row(ui.input_action_button("reset_m2", "Reset M2"))),
                      ui.column(2, ui.row(ui.output_text("m3_text")),
                                ui.row(ui.input_action_button("set_m3", "M3")),
                                ui.row(ui.input_action_button("reset_m3", "Reset M3"))),
                      ui.column(2, ui.row(ui.output_text("m4_text")),
                                ui.row(ui.input_action_button("set_m4", "M4")),
                                ui.row(ui.input_action_button("reset_m4", "Reset M4"))),
                ui.row(
                    ui.column(4, ui.input_selectize("custom_column", 'Select custom column', choices=list_custom_columns)),
                    ui.column(4, ui.input_text("new_custom_column", "Insert new custom column")),
                    ui.column(2, ui.input_action_button("custom_annotation", "Annotate"), ui.input_action_button("reset_custom_annotation", "Reset")),
                ),
                ui.row(ui.input_slider("volume_plot_size", "Volume plot size", min=300, max=1000, step=10, value=400)),
                  align='center'), align='center')

timepoint_selector = ui.column(5,
                  ui.output_plot("plot", height='60vh'),
                  ui.row(
                      ui.column(4,
                                ui.input_action_button(
                                    "previous_time", "previous time", height='15vh')),
                      ui.column(4, ui.input_selectize(
                          "time", 'Select time', choices=times)),
                      ui.column(4, ui.input_action_button(
                          "next_time", "next time", height='15vh')),
                  ),
                  ui.row(
                      ui.column(4,
                                ui.input_action_button(
                                    "previous_point", "previous point")),
                      ui.column(4, ui.input_selectize(
                          "point", 'Select point', choices=points)),
                      ui.column(4, ui.input_action_button(
                          "next_point", "next point")),
                  ),
                  ui.row(ui.input_selectize("channel", 'Select channel', choices=list_channels)),
                  ui.row(ui.input_selectize("column_to_plot", 'Select column to plot', choices=filemap.columns.tolist())))

app_ui = ui.page_fluid(
    ui.row(
        molt_annotator,
        timepoint_selector
    ))


def set_marker_shape(selected_time, worm_types, hatch_time, m1, m2, m3, m4, custom_annotations: list = []):
    symbols = []
    for worm_type in worm_types:
        if worm_type == "egg":
            symbol = "square-open"
        elif worm_type == "worm":
            symbol = "circle-open"
        else:
            symbol = "triangle-up-open"
        symbols.append(symbol)

    # create a list full of "blue"
    sizes = [4] * len(worm_types)
    colors = ["black"] * len(worm_types)

    for custom_annotation in custom_annotations:
        if np.isfinite(custom_annotation):
            symbols[int(custom_annotation)] = "circle"
            sizes[int(custom_annotation)] = 8
            colors[int(custom_annotation)] = "pink"

    if np.isfinite(hatch_time):
        symbols[int(hatch_time)] = "square"
        sizes[int(hatch_time)] = 8
        colors[int(hatch_time)] = "red"

    if np.isfinite(m1):
        symbols[int(m1)] = "circle"
        sizes[int(m1)] = 8
        colors[int(m1)] = "orange"

    if np.isfinite(m2):
        symbols[int(m2)] = "circle"
        sizes[int(m2)] = 8
        colors[int(m2)] = "yellow"

    if np.isfinite(m3):
        symbols[int(m3)] = "circle"
        sizes[int(m3)] = 8
        colors[int(m3)] = "green"

    if np.isfinite(m4):
        symbols[int(m4)] = "circle"
        sizes[int(m4)] = 8
        colors[int(m4)] = "blue"

    widths = [1] * len(worm_types)
    widths[int(selected_time)] = 4
    markers = dict(symbol=symbols, size=sizes,
                   color=colors, line=dict(width=widths))
    return markers


def server(input, output, session):

    hatch = reactive.Value('')
    m1 = reactive.Value('')
    m2 = reactive.Value('')
    m3 = reactive.Value('')
    m4 = reactive.Value('')

    @ reactive.Calc
    def get_images_of_point():
        images_of_point_paths = filemap[filemap['Point'] == int(
            input.point())]['raw'].values.tolist()
        images_of_point = images_of_point_paths
        # images_of_point = Parallel(n_jobs=-1)(delayed(image_handling.read_tiff_file)(path) for path in images_of_point_paths['raw'].values.tolist())
        return images_of_point
    
    def get_custom_annotations():
        if input.custom_column() == "":
            return []
        custom_annotations = filemap[filemap['Point'] == int(input.point())][input.custom_column()].values.tolist()
        return custom_annotations
    
    @ reactive.Effect
    def get_hatch_and_molts():
        hatch_and_molts = filemap[filemap['Point'] == int(input.point())][['HatchTime', 'M1', 'M2', 'M3', 'M4']].values.tolist()[0]

        hatch.set(hatch_and_molts[0])
        m1.set(hatch_and_molts[1])
        m2.set(hatch_and_molts[2])
        m3.set(hatch_and_molts[3])
        m4.set(hatch_and_molts[4])

    @ reactive.Effect
    @ reactive.event(input.previous_time)
    def previous_time():
        print("previous_time")
        new_time = max(int(input.time()) - 1, 0)
        ui.update_selectize('time', selected=str(int(new_time)))

    @ reactive.Effect
    @ reactive.event(input.next_time)
    def next_time():
        print("next_time")
        new_time = min(int(input.time()) + 1, np.max(times) - 1)
        ui.update_selectize('time', selected=str(int(new_time)))

    @ reactive.Effect
    @ reactive.event(input.previous_point)
    def previous_point():
        print("previous_point")
        new_point = max(int(input.point()) - 1, 0)
        ui.update_selectize('point', selected=str(int(new_point)))

    @ reactive.Effect
    @ reactive.event(input.next_point)
    def next_point():
        print("next_point")
        new_point = min(int(input.point()) + 1, np.max(points) - 1)
        ui.update_selectize('point', selected=str(int(new_point)))

    @ output
    @ render_widget
    def volume_plot():
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), ['Time', input.column_to_plot(), 'WormType', 'HatchTime', 'M1', 'M2', 'M3', 'M4']]
        data_of_point[input.column_to_plot()] = np.log10(data_of_point[input.column_to_plot()])
        markers = set_marker_shape(input.time(),
                                   data_of_point['WormType'], hatch(), m1(), m2(), m3(), m4(), custom_annotations=get_custom_annotations())
        fig = go.FigureWidget()
        fig.add_trace(go.Scatter(
            x=data_of_point['Time'], y=data_of_point[input.column_to_plot()], mode="markers", marker=markers))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=input.column_to_plot(),
            legend_title="Worm Type",
            margin=dict(l=20, r=20, t=50, b=50),
            height=input.volume_plot_size(),
        )

        def update_selected_time(trace, points, selector):
            print("update_selected_time")
            for point in points.point_inds:
                ui.update_selectize('time', selected=str(point))

        fig.data[0].on_click(update_selected_time)

        return fig
    
    @ output
    @ render.text
    def hatch_text():
        return f"Hatch : {str(hatch())}"
    
    @ output
    @ render.text
    def m1_text():
        return f"M1 : {str(m1())}"
    
    @ output
    @ render.text
    def m2_text():
        return f"M2: {str(m2())}"
    
    @ output
    @ render.text
    def m3_text():
        return f"M3 : {str(m3())}"

    @ output
    @ render.text
    def m4_text():
        return f"M4 : {str(m4())}"
    
    @ reactive.Effect
    @ reactive.event(input.set_hatch)
    def set_hatch():
        print("set_hatch") 
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), [input.column_to_plot(), 'WormType']]
        volume = data_of_point[input.column_to_plot()].values
        worm_types = data_of_point['WormType'].values
        new_hatch = float(input.time())
        filemap.loc[filemap['Point'] == int(input.point()), ['HatchTime']] = new_hatch
        volume_at_hatch = compute_volume_at_time(volume, worm_types, new_hatch)
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtHatch']] = volume_at_hatch
        filemap.to_csv(filemap_path, index=False)
        hatch.set(new_hatch)

    @ reactive.Effect
    @ reactive.event(input.set_m1)
    def set_m1():
        print("set_m1")
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), [input.column_to_plot(), 'WormType']]
        volume = data_of_point[input.column_to_plot()].values
        worm_types = data_of_point['WormType'].values
        new_m1 = float(input.time())
        filemap.loc[filemap['Point'] == int(input.point()), ['M1']] = new_m1
        volume_at_new_m1 = compute_volume_at_time(volume, worm_types, new_m1)
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM1']] = volume_at_new_m1
        filemap.to_csv(filemap_path, index=False)
        m1.set(new_m1)

    @ reactive.Effect
    @ reactive.event(input.set_m2)
    def set_m2():
        print("set_m2")
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), [input.column_to_plot(), 'WormType']]
        volume = data_of_point[input.column_to_plot()].values
        worm_types = data_of_point['WormType'].values
        new_m2 = float(input.time())
        filemap.loc[filemap['Point'] == int(input.point()), ['M2']] = new_m2
        volume_at_new_m2 = compute_volume_at_time(volume, worm_types, new_m2)
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM2']] = volume_at_new_m2
        filemap.to_csv(filemap_path, index=False)
        m1.set(new_m2)

    @ reactive.Effect
    @ reactive.event(input.set_m3)
    def set_m3():
        print("set_m3")
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), [input.column_to_plot(), 'WormType']]
        volume = data_of_point[input.column_to_plot()].values
        worm_types = data_of_point['WormType'].values
        new_m3 = float(input.time())
        filemap.loc[filemap['Point'] == int(input.point()), ['M3']] = new_m3
        volume_at_new_m3 = compute_volume_at_time(volume, worm_types, new_m3)
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM3']] = volume_at_new_m3
        filemap.to_csv(filemap_path, index=False)
        m3.set(new_m3)

    @ reactive.Effect
    @ reactive.event(input.set_m4)
    def set_m4():
        print("set_m4")
        data_of_point = filemap.loc[filemap['Point'] == int(
            input.point()), [input.column_to_plot(), 'WormType']]
        volume = data_of_point[input.column_to_plot()].values
        worm_types = data_of_point['WormType'].values
        new_m4 = float(input.time())
        filemap.loc[filemap['Point'] == int(input.point()), ['M4']] = new_m4
        volume_at_new_m4 = compute_volume_at_time(volume, worm_types, new_m4)
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM4']] = volume_at_new_m4
        filemap.to_csv(filemap_path, index=False)
        m4.set(new_m4)


    @ reactive.Effect
    @ reactive.event(input.reset_hatch)
    def reset_hatch():
        print("reset_hatch")
        filemap.loc[filemap['Point'] == int(input.point()), ['HatchTime']] = np.nan
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtHatch']] = np.nan
        filemap.to_csv(filemap_path, index=False)
        hatch.set(np.nan)

    
    @ reactive.Effect
    @ reactive.event(input.reset_m1)
    def reset_m1():
        print("reset_m1")
        filemap.loc[filemap['Point'] == int(input.point()), ['M1']] = np.nan
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM1']] = np.nan
        filemap.to_csv(filemap_path, index=False)
        m1.set(np.nan)

    @ reactive.Effect
    @ reactive.event(input.reset_m2)
    def reset_m2():
        print("reset_m2")
        filemap.loc[filemap['Point'] == int(input.point()), ['M2']] = np.nan
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM2']] = np.nan
        filemap.to_csv(filemap_path, index=False)
        m2.set(np.nan)

    @ reactive.Effect
    @ reactive.event(input.reset_m3)
    def reset_m3():
        print("reset_m3")
        filemap.loc[filemap['Point'] == int(input.point()), ['M3']] = np.nan
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM3']] = np.nan
        filemap.to_csv(filemap_path, index=False)
        m3.set(np.nan)

    @ reactive.Effect
    @ reactive.event(input.reset_m4)
    def reset_m4():
        print("reset_m4")
        filemap.loc[filemap['Point'] == int(input.point()), ['M4']] = np.nan
        filemap.loc[filemap['Point'] == int(input.point()), ['VolumeAtM4']] = np.nan
        filemap.to_csv(filemap_path, index=False)
        m4.set(np.nan)

    @ reactive.Effect
    @ reactive.event(input.custom_annotation)
    def custom_annotation():
        new_column_name = input.new_custom_column()
        custom_column = input.custom_column()

        if new_column_name != "" and (new_column_name not in list_custom_columns):
            print("new column")
            list_custom_columns.append(new_column_name)
            ui.update_selectize('custom_column', choices=list_custom_columns, selected=new_column_name)
            ui.update_text('new_custom_column', value="")
            filemap[new_column_name] = np.nan
            filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time()))), new_column_name] = float(input.time())
            filemap.to_csv(filemap_path, index=False)

            print(filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time())))])

        if new_column_name == "" and custom_column != "":
            print("add custom annotation")
            filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time()))), custom_column] = float(input.time())
            filemap.to_csv(filemap_path, index=False)

            print(filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time())))])
            
    @ reactive.Effect
    @ reactive.event(input.reset_custom_annotation)
    def reset_custom_annotation():
        custom_column = input.custom_column()
        if custom_column != "":
            print("reset custom annotation")
            filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time()))), custom_column] = np.nan
            filemap.to_csv(filemap_path, index=False)

            print(filemap.loc[((filemap['Point'] == int(input.point())) & (filemap['Time'] == int(input.time())))])

    
    @ output
    @ render.plot
    def plot():

        channel = input.channel()
        channel = channel.split(' ')[-1]
        channel = int(channel) - 1

        images_of_point = get_images_of_point()
        img = images_of_point[int(input.time())]
        img = image_handling.read_tiff_file(img)
        print(img.shape)

        fig, ax = plt.subplots()
        im = ax.imshow(img[channel].squeeze(), cmap='gray')
        return fig


app = App(app_ui, server)