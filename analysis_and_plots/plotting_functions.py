import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from towbintools.data_analysis import rescale_and_aggregate, filter_series_with_classification
from itertools import combinations
from scipy.stats import mannwhitneyu
import pandas as pd
import starbars


def plot_aggregated_series(conditions_struct, series_column, conditions_to_plot, experiment_time = True, aggregation='mean', n_points=100, time_step = 10, log_scale = True, color_palette = "colorblind"):
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

            plt.plot(rescaled_time, aggregated_series, color=color_palette[i], label=f'Condition {condition_id}')
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
    plt.show()

def plot_correlation(conditions_struct, column_one, column_two, conditions_to_plot,  log_scale = True, color_palette = "colorblind"):
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

        plt.plot(aggregated_series_one, aggregated_series_two, color=color_palette[i], label=f'Condition {condition_id}')




    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.yscale('log' if log_scale else 'linear')
    plt.xscale('log' if log_scale else 'linear')

    plt.legend()
    plt.show()

def plot_correlation_at_molt(conditions_struct, column_one, column_two, conditions_to_plot, log_scale = True, color_palette = "colorblind"):
    color_palette = sns.color_palette(color_palette, len(conditions_to_plot))
    
    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        x = np.nanmean(condition_dict[column_one], axis=0)
        x_std = np.nanstd(condition_dict[column_one], axis=0)
        x_ste = x_std / np.sqrt(np.sum(np.isfinite(condition_dict[column_one]), axis=0))

        y = np.nanmean(condition_dict[column_two], axis=0)
        y_std = np.nanstd(condition_dict[column_two], axis=0)
        y_ste = y_std / np.sqrt(np.sum(np.isfinite(condition_dict[column_two]), axis=0))

        plt.errorbar(x, y, xerr=x_std, yerr=y_std, fmt='o', color=color_palette[i], label=f'Condition {condition_id}', capsize=3)
        plt.plot(x, y, color=color_palette[i])
        
    plt.xlabel(column_one)
    plt.ylabel(column_two)
    plt.yscale('log' if log_scale else 'linear')
    plt.xscale('log' if log_scale else 'linear')

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
                    column: np.log(value) if log_scale else value
                })
   
    df = pd.DataFrame(data_list)
   
    # Determine figure size
    if figsize is None:
        figsize = (5 * df['Molt'].nunique(), 6)
   
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
            dodge=False,
        )

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
    ax[0].set_ylabel(column)

    # remove x label
   
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

def plot_growth_curves_individuals(conditions_struct, column, conditions_to_plot, log_scale = True, color_palette = "colorblind", figsize = None):
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

            ax[i].plot(time, filtered_data)

        ax[i].set_yscale('log' if log_scale else 'linear')
        
    ax[0].set_ylabel(column)

    plt.show()

def get_proportion_model(series_one_at_ecdysis, series_two_at_ecdysis, remove_hatch = True):
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
    plt.xlabel('column one')
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

def plot_deviation_from_model_at_ecdysis(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, remove_hatch = True):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))
    control_condition = conditions_struct[control_condition_id]
    control_model = get_proportion_model(control_condition[column_one], control_condition[column_two], remove_hatch)

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        body_data, pharynx_data = condition[column_one], condition[column_two]
        x, y, y_err = get_deviation_from_model_at_ecdysis(body_data, pharynx_data, control_model, remove_hatch)

        plt.plot(x, y, label=f'condition {condition_id}', color = color_palette[i], marker='o')
        plt.errorbar(x, y, yerr=y_err, color=color_palette[i], fmt='o', capsize=3)

    plt.legend()
    plt.show()


def plot_normalized_proportions(conditions_struct, column_one, column_two, control_condition_id, conditions_to_plot, aggregation='mean', log_scale = True):
    color_palette = sns.color_palette("husl", len(conditions_to_plot))
    control_condition = conditions_struct[control_condition_id]
    control_column_one, control_column_two = control_condition[column_one], control_condition[column_two]

    aggregation_function = np.nanmean
    control_proportion = aggregation_function(control_column_two / control_column_one, axis=0)

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        condition_column_one = condition[column_one]
        condition_column_two = condition[column_two]

        proportion = condition_column_two / condition_column_one
        normalized_proportion = proportion / control_proportion

        y = aggregation_function(normalized_proportion, axis=0)
        y_err = np.nanstd(normalized_proportion, axis=0)/np.sqrt(len(normalized_proportion))
        x = aggregation_function(condition_column_one, axis=0)

        plt.plot(x, y, label=f'condition {condition_id}', color=color_palette[i])
        plt.errorbar(x, y, yerr=y_err, fmt='o', capsize=3, color=color_palette[i])
    
    plt.legend()
    plt.xlabel(column_one)
    plt.ylabel(f'normalized {column_two} to {column_one} ratio')
    plt.xscale('log' if log_scale else 'linear')
    plt.show()