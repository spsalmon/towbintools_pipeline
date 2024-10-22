import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from towbintools.data_analysis import rescale_and_aggregate, filter_series_with_classification, rescale_series, aggregate_interpolated_series
from itertools import combinations
from scipy.stats import mannwhitneyu
import pandas as pd
import starbars
from scipy.interpolate import make_interp_spline
import matplotlib.patches as mpatches
import statsmodels.api as sm
import yaml
import os
from towbintools.data_analysis import compute_growth_rate_per_larval_stage, correct_series_with_classification, filter_series_with_classification, compute_larval_stage_duration, rescale_and_aggregate, compute_series_at_time_classified
from collections import defaultdict

# BUILDING THE PLOTTING STRUCTURE

def build_conditions(config):
    conditions = []
    condition_id = 0

    for condition in config["conditions"]:
        condition = {key: [val] if not isinstance(val, list) else val for key, val in condition.items()}

        lengths = set(len(val) for val in condition.values())
        if len(lengths) > 2 or (len(lengths) == 2 and 1 not in lengths):
            raise ValueError("All lists in the condition must have the same length or be of length 1.")

        max_length = max(lengths)
        for i in range(max_length):
            condition_dict = {
                key: val[0] if len(val) == 1 else val[i]
                for key, val in condition.items()
            }
            condition_dict["condition_id"] = condition_id
            conditions.append(condition_dict)
            condition_id += 1

    return conditions

def add_conditions_to_filemap(experiment_filemap, conditions, config):
    for condition in conditions:
        if "point_range" in condition.keys():
            point_range = condition["point_range"]
            
            # check if point range is a list of lists
            if isinstance(point_range[0], list):
                for pr in point_range:
                    # Get all the rows that are in the point range
                    condition_rows = experiment_filemap[experiment_filemap["Point"].between(pr[0], pr[1])]
                    # Remove the point range from the condition
                    conditions_to_add = {key: val for key, val in condition.items() if key != "point_range"}
                    for key, val in conditions_to_add.items():
                        # Directly fill the rows with the value for the new or existing column
                        experiment_filemap.loc[condition_rows.index, key] = val
            else:
                # Get all the rows that are in the point range
                condition_rows = experiment_filemap[experiment_filemap["Point"].between(point_range[0], point_range[1])]
                # Remove the point range from the condition
                conditions_to_add = {key: val for key, val in condition.items() if key != "point_range"}
                for key, val in conditions_to_add.items():
                    # Directly fill the rows with the value for the new or existing column
                    experiment_filemap.loc[condition_rows.index, key] = val

        elif "pad" in condition.keys():
            pad = condition["pad"]
            # Get all the rows that are in the pad
            condition_rows = experiment_filemap[experiment_filemap["Pad"] == pad]
            # Remove the pad from the condition
            conditions_to_add = {key: val for key, val in condition.items() if key != "pad"}
            for key, val in conditions_to_add.items():
                # Directly fill the rows with the value for the new or existing column
                experiment_filemap.loc[condition_rows.index, key] = val
                
        else:
            print("Condition does not contain 'point_range' or 'pad' key, impossible to add condition to filemap, skipping.")
    return experiment_filemap

def get_ecdysis_and_durations(filemap):
    all_ecdysis_time_step = []
    all_durations_time_step = []

    all_ecdysis_experiment_time = []
    all_durations_experiment_time = []
    
    for point in filemap["Point"].unique():
        point_df = filemap[filemap["Point"] == point]
        point_ecdysis = point_df[["HatchTime", "M1", "M2", "M3", "M4"]].iloc[0]
        larval_stage_durations = list(compute_larval_stage_duration(point_ecdysis).values())

        point_ecdysis = point_ecdysis.to_numpy()
        all_ecdysis_time_step.append(point_ecdysis)
        all_durations_time_step.append(larval_stage_durations)

        ecdysis_experiment_time = []
        for ecdys in point_ecdysis:
            if np.isnan(ecdys):
                ecdysis_experiment_time.append(np.nan)
            else:
                ecdys_experiment_time = point_df[point_df["Time"] == ecdys]["ExperimentTime"].iloc[0]
                ecdysis_experiment_time.append(ecdys_experiment_time)

        all_ecdysis_experiment_time.append(ecdysis_experiment_time)
        
        durations_experiment_time = []
        for i in range(len(ecdysis_experiment_time) - 1):
            start = ecdysis_experiment_time[i]
            end = ecdysis_experiment_time[i + 1]
            duration_experiment_time = end - start
            durations_experiment_time.append(duration_experiment_time)

        all_durations_experiment_time.append(durations_experiment_time)
        
    return np.array(all_ecdysis_time_step), np.array(all_durations_time_step), np.array(all_ecdysis_experiment_time), np.array(all_durations_experiment_time)

def separate_column_by_point(filemap, column):
    separated_column = []
    for point in filemap["Point"].unique():
        point_df = filemap[filemap["Point"] == point]
        separated_column.append(point_df[column].values)
    return np.array(separated_column)

def build_plotting_struct(experiment_dir, filemap_path, config_path, organ_channels = {'body' : 2, 'pharynx' : 1}):

    experiment_filemap = pd.read_csv(filemap_path)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        file.close()

    conditions = build_conditions(config)
    conditions_keys = list(conditions[0].keys())

    # remove 'point_range' and 'pad' from the conditions keys if they are present
    if 'point_range' in conditions_keys:
        conditions_keys.remove('point_range')
    if 'pad' in conditions_keys:
        conditions_keys.remove('pad')

    experiment_filemap = add_conditions_to_filemap(experiment_filemap, conditions, config)

    experiment_filemap.columns

    # if ExperimentTime is not present in the filemap, add it
    if 'ExperimentTime' not in experiment_filemap.columns:
        experiment_filemap['ExperimentTime'] = np.nan

    # remove rows where condition_id is NaN
    experiment_filemap = experiment_filemap[~experiment_filemap['condition_id'].isnull()]

    conditions_struct = []
        
    for condition_id in experiment_filemap["condition_id"].unique():
        condition_df = experiment_filemap[experiment_filemap["condition_id"] == condition_id]
        condition_dict = {}
        for key in conditions_keys:
            condition_dict[key] = condition_df[key].iloc[0]

        ecdysis_time_step, larval_stage_durations_time_step, ecdysis_experiment_time, larval_stage_durations_experiment_time = get_ecdysis_and_durations(condition_df)
        condition_dict['condition_id'] = int(condition_dict['condition_id'])
        condition_dict['ecdysis_time_step'] = ecdysis_time_step
        condition_dict['larval_stage_durations_time_step'] = larval_stage_durations_time_step
        condition_dict['ecdysis_experiment_time'] = ecdysis_experiment_time
        condition_dict['larval_stage_durations_experiment_time'] = larval_stage_durations_experiment_time
        condition_dict['experiment'] = experiment_dir

        worm_type_column = [col for col in condition_df.columns if 'worm_type' in col][0]
        worm_types = separate_column_by_point(condition_df, worm_type_column)

        for organ in organ_channels.keys():
            organ_channel = organ_channels[organ]
            organ_channel = f'ch{organ_channel}'
            organ_columns = [col for col in condition_df.columns if col.startswith(organ_channel)]
            organ_columns = [col for col in organ_columns if not ('_at_' in col)]
            renamed_organ_columns = [col.replace(organ_channel, organ) for col in organ_columns]

            for organ_column, renamed_organ_column in zip(organ_columns, renamed_organ_columns):
                condition_dict[renamed_organ_column] = separate_column_by_point(condition_df, organ_column)

            # remove any column with worm_type in it
            renamed_organ_columns = [col for col in renamed_organ_columns if not ('worm_type' in col)]
            for column in renamed_organ_columns:
                condition_dict[f'{column}_at_ecdysis'] = np.stack([compute_series_at_time_classified(condition_dict[column][i], worm_types[i], ecdysis_time_step[i]) for i in range(len(ecdysis_time_step))])


        condition_dict['time'] = separate_column_by_point(condition_df, 'Time').astype(float)
        condition_dict['experiment_time'] = separate_column_by_point(condition_df, 'ExperimentTime').astype(float)

        conditions_struct.append(condition_dict)

    conditions_info = [{key : condition[key] for key in conditions_keys} for condition in conditions_struct]

    # sort the conditions and conditions_info by condition_id
    conditions_struct = sorted(conditions_struct, key=lambda x: x['condition_id'])
    conditions_info = sorted(conditions_info, key=lambda x: x['condition_id'])

    return conditions_struct, conditions_info


def remove_unwanted_info(conditions_info):
    for condition in conditions_info:
        if 'description' in condition.keys():
            condition.pop('description')
        if 'condition_id' in condition.keys():
            condition.pop('condition_id')
    return conditions_info

# def combine_experiments(filemap_paths, config_paths, experiment_dirs = None):
#     condition_info_merge_list = []
#     conditions_info_keys = []
#     all_conditions_struct = []
#     for i, (filemap_path, config_path) in enumerate(zip(filemap_paths, config_paths)):
#         if experiment_dirs is not None:
#             experiment_dir = experiment_dirs[i]
#         else:
#             experiment_dir = os.path.dirname(filemap_path)
#         conditions_struct, conditions_info = build_plotting_struct(experiment_dir, filemap_path, config_path)

#         # if all_conditions_struct is empty, just add the conditions
#         if not all_conditions_struct:
#             all_conditions_struct.extend(conditions_struct)
#         # else, extend the conditions but modify the condition_id to be unique
#         else:
#             max_condition_id = max([condition['condition_id'] for condition in all_conditions_struct]) + 1
#             print(f'Max condition id: {max_condition_id}')
#             for condition in conditions_struct:
#                 condition['condition_id'] += max_condition_id
#                 all_conditions_struct.append(condition)

#         experiment_conditions_info_keys = [list(conditions_info[i].keys()) for i in range(len(conditions_info))]
#         experiment_conditions_info_keys = [item for sublist in experiment_conditions_info_keys for item in sublist]
#         conditions_info_keys.extend(experiment_conditions_info_keys)
#         conditions_info_merge = remove_unwanted_info(conditions_info)

#         condition_info_merge_list.extend(conditions_info_merge)
#     # merge conditions that have the exact same info
#     conditions_to_merge = []
#     for i, condition_info in enumerate(condition_info_merge_list):
#         merge_with = []
#         for j, other_condition in enumerate(condition_info_merge_list):
#             if condition_info == other_condition and i != j:
#                 merge_with.append(j)
#         if merge_with:
#             conditions_to_merge.append([i] + merge_with)
#         else:
#             conditions_to_merge.append([i])

#     # remove permutations
#     conditions_to_merge = [sorted(merge) for merge in conditions_to_merge]
#     conditions_to_merge = list(set([tuple(merge) for merge in conditions_to_merge]))
#     conditions_to_merge = [list(merge) for merge in conditions_to_merge]
#     # sort the list of conditions to merge by the first element
#     conditions_to_merge = sorted(conditions_to_merge, key=lambda x: x[0])

#     print(conditions_to_merge)

#     merged_conditions_struct = []

#     conditions_info_keys = list(set(conditions_info_keys))
#     fields_not_to_merge = conditions_info_keys

#     for merge in conditions_to_merge:
#         if len(merge) == 1:
#             merged_conditions_struct.append(all_conditions_struct[merge[0]])
#         else:
#             base_condition = all_conditions_struct[merge[0]]
#             for condition_index in merge[1:]:
#                 for key in base_condition.keys():
#                     if key not in fields_not_to_merge:
#                         base_condition_data = base_condition[key]
#                         condition_data = all_conditions_struct[condition_index][key]

#                         if isinstance(base_condition_data, np.ndarray):
#                             # get the shortest dimension
#                             smallest = np.argmin([base_condition_data.shape[1], condition_data.shape[1]])
#                             # add nan to the condition data to match the biggest dimension
#                             if smallest == 0:
#                                 base_condition_data = np.pad(base_condition_data, ((0, 0), (0, condition_data.shape[1] - base_condition_data.shape[1])), mode='constant', constant_values=np.nan)
#                             else:
#                                 condition_data = np.pad(condition_data, ((0, 0), (0, base_condition_data.shape[1] - condition_data.shape[1])), mode='constant', constant_values=np.nan)
#                         try:
#                             base_condition[key] = np.concatenate((base_condition_data, condition_data), axis=0)
#                         except ValueError as e:
#                             print(f"Could not concatenate {key} : {e}")

#                 merged_conditions_struct.append(base_condition)

#     # sort the merged conditions by condition_id
#     merged_conditions_struct = sorted(merged_conditions_struct, key=lambda x: x['condition_id'])

#     # remove gaps in the condition_id
#     for i in range(len(merged_conditions_struct)):
#         merged_conditions_struct[i]['condition_id'] = i

#     return merged_conditions_struct

def combine_experiments(filemap_paths, config_paths, experiment_dirs=None):
    all_conditions_struct = []
    condition_info_merge_list = []
    conditions_info_keys = set()
    condition_id_counter = 0

    # Process each experiment
    for i, (filemap_path, config_path) in enumerate(zip(filemap_paths, config_paths)):
        experiment_dir = experiment_dirs[i] if experiment_dirs else os.path.dirname(filemap_path)
        conditions_struct, conditions_info = build_plotting_struct(experiment_dir, filemap_path, config_path)

        # Process conditions for this experiment
        for condition in conditions_struct:
            condition['condition_id'] = condition_id_counter
            condition_id_counter += 1
            all_conditions_struct.append(condition)

        # Process condition info
        experiment_conditions_info = remove_unwanted_info(conditions_info)
        condition_info_merge_list.extend(experiment_conditions_info)
        conditions_info_keys.update(*[condition.keys() for condition in experiment_conditions_info])

    # Merge conditions based on their info
    condition_dict = defaultdict(list)
    for i, condition_info in enumerate(condition_info_merge_list):
        key = frozenset(condition_info.items())
        condition_dict[key].append(i)

    merged_conditions_struct = []
    for indices in condition_dict.values():
        base_condition = all_conditions_struct[indices[0]]
        for idx in indices[1:]:
            for key, value in all_conditions_struct[idx].items():
                if key not in conditions_info_keys:
                    if isinstance(value, np.ndarray):
                        if value.shape[1] > base_condition[key].shape[1]:
                            base_condition[key] = np.pad(base_condition[key], 
                                ((0, 0), (0, value.shape[1] - base_condition[key].shape[1])), 
                                mode='constant', constant_values=np.nan)
                        elif value.shape[1] < base_condition[key].shape[1]:
                            value = np.pad(value, 
                                ((0, 0), (0, base_condition[key].shape[1] - value.shape[1])), 
                                mode='constant', constant_values=np.nan)
                    try:
                        base_condition[key] = np.concatenate((base_condition[key], value), axis=0)
                    except ValueError as e:
                        print(f"Could not concatenate {key}: {e}")

        merged_conditions_struct.append(base_condition)

    # # Sort and reassign condition IDs
    # merged_conditions_struct.sort(key=lambda x: x['condition_id'])
    for i, condition in enumerate(merged_conditions_struct):
        condition['condition_id'] = i

    return merged_conditions_struct

# PLOTTING FUNCTIONS

def build_legend(single_condition_dict, legend):
    if legend is None:
        return f'Condition {int(single_condition_dict["condition_id"])}'
    else:
        legend_string = ''
        for i, (key, value) in enumerate(legend.items()):
            if value:
                legend_string += f'{single_condition_dict[key]} {value}'
            else:
                legend_string += f'{single_condition_dict[key]}'
            if i < len(legend) - 1:
                legend_string += ', '
        return legend_string

def set_scale(ax, log_scale):
    if isinstance(log_scale, bool):
        ax.set_yscale('log' if log_scale else 'linear')
    elif isinstance(log_scale, tuple):
        ax.set_yscale('log' if log_scale[1] else 'linear')
        ax.set_xscale('log' if log_scale[0] else 'linear')
    elif isinstance(log_scale, list):
        ax.set_yscale('log' if log_scale[1] else 'linear')
        ax.set_xscale('log' if log_scale[0] else 'linear')

def plot_aggregated_series(conditions_struct, series_column, conditions_to_plot, experiment_time = True, aggregation='mean', n_points=100, time_step = 10, log_scale = True, colors = None, legend = None, y_axis_label = None):
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    def plot_single_series(column: str):
        for i, condition_id in enumerate(conditions_to_plot):
            condition_dict = conditions_struct[condition_id]
            if experiment_time:
                time = condition_dict['experiment_time']
                larval_stage_durations = condition_dict['larval_stage_durations_experiment_time']
            else:
                time = condition_dict['time']
                larval_stage_durations = condition_dict['larval_stage_durations_time_step']
            
            rescaled_time, aggregated_series, _, ste_series = rescale_and_aggregate(
                condition_dict[column], 
                time, 
                condition_dict['ecdysis_time_step'], 
                larval_stage_durations, 
                condition_dict['body_seg_str_worm_type'], 
                aggregation=aggregation
            )
            

            ci_lower = aggregated_series - 1.96*ste_series
            ci_upper = aggregated_series + 1.96*ste_series

            if experiment_time:
                rescaled_time = rescaled_time / 3600
            else:
                rescaled_time = rescaled_time * time_step / 60

            label = build_legend(condition_dict, legend)
                
            plt.plot(rescaled_time, aggregated_series, color=color_palette[i], label=label)
            plt.fill_between(rescaled_time, ci_lower, ci_upper, color=color_palette[i], alpha=0.2)


    if isinstance(series_column, list):
        for column in series_column:
            plot_single_series(column)
    else:
        plot_single_series(series_column)

    # remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel('Time (h)')
    plt.yscale('log' if log_scale else 'linear')

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(series_column)

    plt.show()

def plot_correlation(conditions_struct, column_one, column_two, conditions_to_plot,  log_scale = True, colors = None, legend = None, x_axis_label = None, y_axis_label = None):
    if colors is None:
        color_palette = sns.color_palette('colorblind', len(conditions_to_plot))
    else:
        color_palette = colors
    
    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        _, aggregated_series_one, _, _ = rescale_and_aggregate(
            condition_dict[column_one], 
            condition_dict['time'], 
            condition_dict['ecdysis_time_step'], 
            condition_dict['larval_stage_durations_time_step'], 
            condition_dict['body_seg_str_worm_type'], 
            aggregation='mean'
        )

        _, aggregated_series_two, _, _ = rescale_and_aggregate(
            condition_dict[column_two], 
            condition_dict['time'], 
            condition_dict['ecdysis_time_step'], 
            condition_dict['larval_stage_durations_time_step'], 
            condition_dict['body_seg_str_worm_type'], 
            aggregation='mean'
            )

        # sort the values
        order = np.argsort(aggregated_series_one)
        aggregated_series_one = aggregated_series_one[order]
        aggregated_series_two = aggregated_series_two[order]

        label = build_legend(condition_dict, legend)

        plt.plot(aggregated_series_one, aggregated_series_two, color=color_palette[i], label=label)

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()

def plot_correlation_at_ecdysis(conditions_struct, column_one, column_two, conditions_to_plot, remove_hatch = True, log_scale = True, colors = None, legend = None, x_axis_label = None, y_axis_label = None):
    if colors is None:
        color_palette = sns.color_palette('colorblind', len(conditions_to_plot))
    else:
        color_palette = colors
    
    for i, condition_id in enumerate(conditions_to_plot):

        condition_dict = conditions_struct[condition_id]

        if remove_hatch:
            column_one_values = condition_dict[column_one][:, 1:]
            column_two_values = condition_dict[column_two][:, 1:]
        else:
            column_one_values = condition_dict[column_one]
            column_two_values = condition_dict[column_two]


        x = np.nanmean(column_one_values, axis=0)
        x_std = np.nanstd(column_one_values, axis=0)
        x_ste = x_std / np.sqrt(np.sum(np.isfinite(column_one_values), axis=0))

        y = np.nanmean(column_two_values, axis=0)
        y_std = np.nanstd(column_two_values, axis=0)
        y_ste = y_std / np.sqrt(np.sum(np.isfinite(column_two_values), axis=0))

        label = build_legend(condition_dict, legend)
        plt.errorbar(x, y, xerr=x_std, yerr=y_std, fmt='o', color=color_palette[i], label=label, capsize=3)
        plt.plot(x, y, color=color_palette[i])
        
    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)
    
    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()

def boxplot_at_molt(
    conditions_struct,
    column,
    conditions_to_plot,
    log_scale: bool = True,
    figsize: tuple = None,
    color_palette = "colorblind",
    plot_significance: bool = False,
    legend = None,
    y_axis_label = None,
    titles = None
):

    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for j in range(data.shape[1]):
            for value in data[:, j]:
                data_list.append({
                    'Condition': condition_id,
                    'Molt': j,
                    column: np.log(value) if log_scale else value,
                })
   
    df = pd.DataFrame(data_list)
   
    # Determine figure size
    if figsize is None:
        figsize = (5 * df['Molt'].nunique(), 6)
   
    if titles is not None and len(titles) != df['Molt'].nunique():
        print('Number of titles does not match the number of ecdysis events.')
        titles = None

    # Create plot
    fig, ax = plt.subplots(1, df['Molt'].nunique(), figsize=figsize)
    for i in range(df['Molt'].nunique()):
        sns.boxplot(
            data=df[df['Molt'] == i],
            x='Condition',
            y=column,
            hue='Condition',
            palette=color_palette,
            showfliers=False,
            ax=ax[i],
            dodge=False
        )

        handles, labels = ax[i].get_legend_handles_labels()
        new_label_list = [build_legend(conditions_struct[condition_id], legend) for condition_id in conditions_to_plot]
        ax[i].legend(handles, new_label_list)

        ylims=ax[i].get_ylim()
        sns.stripplot(
            data=df[df['Molt'] == i],
            x='Condition',
            y=column,
            ax=ax[i],
            alpha=0.5,
            color='black',
            dodge=True,
        )
        ax[i].set_ylim(ylims)

        ax[i].set_xlabel('')
        ax[i].set_ylabel('')
        if titles is not None:
            ax[i].set_title(titles[i])
        # remove ticks
        ax[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        if plot_significance:
            pairs = list(combinations(df['Condition'].unique(), 2))
            p_values = []

            bars = []
            for pair in pairs:
                data1 = df[(df['Condition'] == pair[0]) & (df['Molt'] == i)][column].dropna()
                data2 = df[(df['Condition'] == pair[1]) & (df['Molt'] == i)][column].dropna()
                if len(data1) == 0 or len(data2) == 0:
                    continue
                p_value = mannwhitneyu(data1, data2).pvalue
                
                # convert condition id to condition index
                bar = [conditions_to_plot.index(pair[0]), conditions_to_plot.index(pair[1]), p_value]
                bars.append(bar)

            starbars.draw_annotation(bars, ax=ax[i])
    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    # remove x label
   
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_growth_curves_individuals(conditions_struct, column, conditions_to_plot, log_scale = True, color_palette = "colorblind", figsize = None, legend = None, y_axis_label = None):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))

    if figsize is None:
        figsize = (len(conditions_to_plot)*8, 10)

    fig, ax = plt.subplots(1, len(conditions_to_plot), figsize=figsize)

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        for j in range(len(condition_dict[column])):
            time = condition_dict['time'][j]
            data = condition_dict[column][j]
            worm_type = condition_dict['body_seg_str_worm_type'][j]
            
            filtered_data = filter_series_with_classification(data, worm_type)

            label = build_legend(condition_dict, legend)

            ax[i].title.set_text(label)
            ax[i].plot(time, filtered_data)

        set_scale(ax[i], log_scale)
        
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    plt.show()

def get_proportion_model(series_one, series_two, worm_type, x_axis_label = None, y_axis_label = None):
    assert len(series_one) == len(series_two), "The two series must have the same length."

    series_one = np.array(series_one).flatten()
    series_two = np.array(series_two).flatten()
    worm_type = np.array(worm_type).flatten()

    series_one = filter_series_with_classification(series_one, worm_type)
    series_two = filter_series_with_classification(series_two, worm_type)

    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one) & ~np.isnan(series_two)
    series_one = series_one[correct_indices]
    series_two = series_two[correct_indices]

    # log transform the data
    series_one = np.log(series_one)
    series_two = np.log(series_two)

    # for duplicate values, take the mean
    unique_series_one = np.unique(series_one)
    unique_series_two = np.array([np.mean(series_two[series_one == value]) for value in unique_series_one])

    series_one = unique_series_one
    series_two = unique_series_two

    plt.scatter(series_one, series_two)
    
    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel('column one')
    
    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel('column two')

    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(series_two, series_one, frac=1./3)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # plt.scatter(lowess_x, lowess_y, color='red')
    # plt.show()

    # interpolate the loess curve
    model = make_interp_spline(lowess_x, lowess_y, k=3)

    x = np.linspace(min(series_one), max(series_one), 100)
    y = model(x)

    plt.plot(x, y, color='red')
    plt.show()

    return model

def get_deviation_from_model(series_one, series_two, time, ecdysis, model, worm_type, n_points=100):

    _, rescaled_series_one = rescale_series(series_one, time, ecdysis, worm_type, n_points=n_points)
    _, rescaled_series_two = rescale_series(series_two, time, ecdysis, worm_type, n_points=n_points)

    # log transform the data
    rescaled_series_one = np.log(rescaled_series_one)
    rescaled_series_two = np.log(rescaled_series_two)

    expected_series_two = model(rescaled_series_one)

    log_residuals = rescaled_series_two - expected_series_two
    residuals = np.exp(log_residuals)

    aggregated_series_one = np.full((4, n_points), np.nan)
    aggregated_residuals = np.full((4, n_points), np.nan)
    std_residuals = np.full((4, n_points), np.nan)
    ste_residuals = np.full((4, n_points), np.nan)

    for i in range(4):
        aggregated_residuals[i, :] = np.nanmean(residuals[:, i, :], axis=0)
        std_residuals[i, :] = np.nanstd(residuals[:, i, :], axis=0)
        ste_residuals[i, :] = std_residuals[i, :] / np.sqrt(np.sum(np.isfinite(residuals[:, i, :]), axis=0))

        aggregated_series_one[i, :] = np.nanmean(np.exp(rescaled_series_one[:, i, :]), axis=0)


    aggregated_series_one = aggregated_series_one.flatten()

    aggregated_residuals = aggregated_residuals.flatten()
    std_residuals = std_residuals.flatten()
    ste_residuals = ste_residuals.flatten()
    
    return aggregated_series_one, aggregated_residuals, std_residuals, ste_residuals

def plot_deviation_from_model(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, log_scale = (True, False), legend = None, x_axis_label = None, y_axis_label = None):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = y_axis_label if y_axis_label is not None else f'deviation from modeled {column_two}'

    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model(control_condition[column_one], control_condition[column_two], control_condition['body_seg_str_worm_type'], x_axis_label = xlbl, y_axis_label = ylbl)

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        ecdysis = condition['ecdysis_time_step']
        time = condition['time']
        body_data, pharynx_data = condition[column_one], condition[column_two]
        x, residuals, std_residuals, ste_residuals = get_deviation_from_model(body_data, pharynx_data, time, ecdysis, control_model, condition['body_seg_str_worm_type'])

        label = build_legend(condition, legend)
        plt.plot(x, residuals, label=label, color = color_palette[i])
        plt.fill_between(x, residuals - 1.96*std_residuals, residuals + 1.96*std_residuals, color=color_palette[i], alpha=0.2)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()

def exclude_arrests_from_series(series_at_ecdysis):
    filtered_series = np.full(series_at_ecdysis.shape, np.nan)
    # keep only a value at one ecdys event if the next one is not nan
    if series_at_ecdysis.shape[0] == 1 or len(series_at_ecdysis.shape) == 1:
        for i in range(len(series_at_ecdysis)):
            if i == len(series_at_ecdysis) - 1:
                filtered_series[i] = series_at_ecdysis[i]
            elif not np.isnan(series_at_ecdysis[i + 1]):
                filtered_series[i] = series_at_ecdysis[i]
        return filtered_series
    else:
        for i in range(series_at_ecdysis.shape[0]):
            for j in range(series_at_ecdysis.shape[1]):
                if j == series_at_ecdysis.shape[1] - 1:
                    filtered_series[i, j] = series_at_ecdysis[i, j]
                elif not np.isnan(series_at_ecdysis[i, j + 1]):
                    filtered_series[i, j] = series_at_ecdysis[i, j]
        return filtered_series

def get_proportion_model_ecdysis(series_one_at_ecdysis, series_two_at_ecdysis, remove_hatch = True, x_axis_label = None, y_axis_label = None, exclude_arrests = False):
    assert len(series_one_at_ecdysis) == len(series_two_at_ecdysis), "The two series must have the same length."

    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)

    series_one_at_ecdysis = np.array(series_one_at_ecdysis).flatten()
    series_two_at_ecdysis = np.array(series_two_at_ecdysis).flatten()
    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one_at_ecdysis) & ~np.isnan(series_two_at_ecdysis)
    series_one_at_ecdysis = series_one_at_ecdysis[correct_indices]
    series_two_at_ecdysis = series_two_at_ecdysis[correct_indices]

    # log transform the data
    series_one_at_ecdysis = np.log(series_one_at_ecdysis)
    series_two_at_ecdysis = np.log(series_two_at_ecdysis)

    plt.scatter(series_one_at_ecdysis, series_two_at_ecdysis)
    
    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel('column one')

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel('column two')

    fit = np.polyfit(series_one_at_ecdysis, series_two_at_ecdysis, 3)
    model = np.poly1d(fit)

    plt.plot(np.sort(series_one_at_ecdysis), model(np.sort(series_one_at_ecdysis)), color='red')
    plt.show()

    return model

def get_deviation_from_model_at_ecdysis(series_one_at_ecdysis, series_two_at_ecdysis, model, remove_hatch = True, exclude_arrests = False):
    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)

    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one_at_ecdysis) & ~np.isnan(series_two_at_ecdysis)
    series_one_at_ecdysis[~correct_indices] = np.nan
    series_two_at_ecdysis[~correct_indices] = np.nan

    # log transform the data
    series_one_at_ecdysis = np.log(series_one_at_ecdysis)
    series_two_at_ecdysis = np.log(series_two_at_ecdysis)

    expected_series_two = model(series_one_at_ecdysis)

    log_residuals = series_two_at_ecdysis - expected_series_two
    residuals = np.exp(log_residuals)

    y = np.nanmean(residuals, axis=0)
    y_err = np.nanstd(residuals, axis=0)/np.sqrt(len(residuals))
    x = np.nanmean(np.exp(series_one_at_ecdysis), axis=0)

    return x, y, y_err

def get_deviation_percentage_from_model_at_ecdysis(series_one_at_ecdysis, series_two_at_ecdysis, model, remove_hatch=True, exclude_arrests=False):
    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

    if exclude_arrests:
        series_one_at_ecdysis = exclude_arrests_from_series(series_one_at_ecdysis)
        series_two_at_ecdysis = exclude_arrests_from_series(series_two_at_ecdysis)
    
    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one_at_ecdysis) & ~np.isnan(series_two_at_ecdysis)
    series_one_at_ecdysis[~correct_indices] = np.nan
    series_two_at_ecdysis[~correct_indices] = np.nan
    
    # Apply the model to the log-transformed series_one to get expected values
    expected_series_two = np.exp(model(np.log(series_one_at_ecdysis)))
    
    # Calculate percentage deviation using real values
    percentage_deviation = (series_two_at_ecdysis - expected_series_two) / expected_series_two * 100
    
    y = np.nanmean(percentage_deviation, axis=0)
    y_err = np.nanstd(percentage_deviation, axis=0) / np.sqrt(np.sum(~np.isnan(percentage_deviation), axis=0))
    x = np.nanmean(series_one_at_ecdysis, axis=0)
    
    return x, y, y_err

def plot_deviation_from_model_at_ecdysis(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, remove_hatch = True, log_scale = (True, False), colors = None, legend = None, x_axis_label = None, y_axis_label = None, percentage = True, exclude_arrests = False):
    
    if colors is None:
        color_palette = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        color_palette = colors

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = y_axis_label if y_axis_label is not None else f'deviation from modeled {column_two}'

    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model_ecdysis(control_condition[column_one], control_condition[column_two], remove_hatch, x_axis_label = xlbl, y_axis_label = ylbl, exclude_arrests = exclude_arrests)

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        body_data, pharynx_data = condition[column_one], condition[column_two]

        if percentage:
            x, y, y_err = get_deviation_percentage_from_model_at_ecdysis(body_data, pharynx_data, control_model, remove_hatch, exclude_arrests)
        else:
            x, y, y_err = get_deviation_from_model_at_ecdysis(body_data, pharynx_data, control_model, remove_hatch, exclude_arrests)

        label = build_legend(condition, legend)
        plt.plot(x, y, label=label, color = color_palette[i], marker='o')
        plt.errorbar(x, y, yerr=y_err, color=color_palette[i], fmt='o', capsize=3)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()


def plot_normalized_proportions(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, aggregation='mean', log_scale = (True, False), legend = None, x_axis_label = None, y_axis_label = None):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))
    control_condition = conditions_struct[control_condition_id]
    control_column_one, control_column_two = control_condition[column_one], control_condition[column_two]

    aggregation_function = np.nanmean
    control_proportion = aggregation_function(control_column_two / control_column_one, axis=0)

    x_axis_label = x_axis_label if x_axis_label is not None else column_one
    y_axis_label = y_axis_label if y_axis_label is not None else f'normalized {column_two} to {column_one} ratio'

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        condition_column_one = condition[column_one]
        condition_column_two = condition[column_two]

        proportion = condition_column_two / condition_column_one
        normalized_proportion = proportion / control_proportion

        y = aggregation_function(normalized_proportion, axis=0)
        y_err = np.nanstd(normalized_proportion, axis=0)/np.sqrt(len(normalized_proportion))
        x = aggregation_function(condition_column_one, axis=0)

        label = build_legend(condition, legend)

        plt.plot(x, y, label=label, color=color_palette[i])
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=3, color=color_palette[i])
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    
    set_scale(plt.gca(), log_scale)

    plt.legend()
    plt.show()