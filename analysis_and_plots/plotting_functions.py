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

def plot_aggregated_series(conditions_struct, series_column, conditions_to_plot, experiment_time = True, aggregation='mean', n_points=100, time_step = 10, log_scale = True, color_palette = "colorblind", legend = None, y_axis_label = None):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    
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

def plot_correlation(conditions_struct, column_one, column_two, conditions_to_plot,  log_scale = True, color_palette = "colorblind", legend = None, x_axis_label = None, y_axis_label = None):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    
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

def plot_correlation_at_molt(conditions_struct, column_one, column_two, conditions_to_plot, log_scale = True, color_palette = "colorblind", legend = None, x_axis_label = None, y_axis_label = None):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    
    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        x = np.nanmean(condition_dict[column_one], axis=0)
        x_std = np.nanstd(condition_dict[column_one], axis=0)
        x_ste = x_std / np.sqrt(np.sum(np.isfinite(condition_dict[column_one]), axis=0))

        y = np.nanmean(condition_dict[column_two], axis=0)
        y_std = np.nanstd(condition_dict[column_two], axis=0)
        y_ste = y_std / np.sqrt(np.sum(np.isfinite(condition_dict[column_two]), axis=0))

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

def get_proportion_model_ecdysis(series_one_at_ecdysis, series_two_at_ecdysis, remove_hatch = True, x_axis_label = None, y_axis_label = None):
    assert len(series_one_at_ecdysis) == len(series_two_at_ecdysis), "The two series must have the same length."

    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]
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

def get_deviation_from_model_at_ecdysis(series_one_at_ecdysis, series_two_at_ecdysis, model, remove_hatch = True):
    if remove_hatch:
        series_one_at_ecdysis = series_one_at_ecdysis[:, 1:]
        series_two_at_ecdysis = series_two_at_ecdysis[:, 1:]

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

def plot_deviation_from_model_at_ecdysis(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, remove_hatch = True, log_scale = (True, False), legend = None, x_axis_label = None, y_axis_label = None):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = y_axis_label if y_axis_label is not None else f'deviation from modeled {column_two}'

    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model_ecdysis(control_condition[column_one], control_condition[column_two], remove_hatch, x_axis_label = xlbl, y_axis_label = ylbl)

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        body_data, pharynx_data = condition[column_one], condition[column_two]
        x, y, y_err = get_deviation_from_model_at_ecdysis(body_data, pharynx_data, control_model, remove_hatch)

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