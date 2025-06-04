from itertools import combinations

import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from statannotations.stats.StatTest import STATTEST_LIBRARY

from .utils_data_processing import rescale_without_flattening
from .utils_plotting import build_legend
from .utils_plotting import get_colors

STATANNOTATIONS_TESTS = STATTEST_LIBRARY.keys()
custom_test = ["Feltz-Miller", "MSLR"]


def _setup_figure(
    df,
    figsize,
    titles,
):
    # Determine figure size
    if figsize is None:
        figsize = (6 * df["Order"].nunique(), 10)
    if titles is not None and len(titles) != df["Order"].nunique():
        print("Number of titles does not match the number of ecdysis events.")
        titles = None

    fig, ax = plt.subplots(
        1,
        df["Order"].nunique(),
        figsize=(figsize[0] + 3, figsize[1]),
        sharey=False,
    )

    return fig, ax


def feltz_miller_asymptotic_cv_test(sample1, sample2):
    """
    Perform the Feltz-Miller asymptotic test for equality of CV on two samples.

    Adapted from: https://github.com/benmarwick/cvequality/blob/master/R/functions.R
    """
    k = 2
    n_j = [len(sample1), len(sample2)]
    s_j = [bn.nanstd(sample1), bn.nanstd(sample2)]
    x_j = [bn.nanmean(sample1), bn.nanmean(sample2)]

    n_j, s_j, x_j = np.array(n_j), np.array(s_j), np.array(x_j)

    m_j = n_j - 1

    D = (np.sum(m_j * (s_j / x_j))) / np.sum(m_j)

    # test statistic
    D_AD = (np.sum(m_j * (s_j / x_j - D) ** 2)) / (D**2 * (0.5 + D**2))

    # D_AD distributes as a Chi-squared distribution with k-1 degrees of freedom
    p_value = 1 - stats.chi2.cdf(D_AD, k - 1)
    return D_AD, p_value


def _LRT_STAT(n, x, s):
    """
    LRT_STAT function required by mslr_test

    Parameters:
    n : array-like
        Sample sizes for each group
    x : array-like
        Means for each group
    s : array-like
        Standard deviations for each group

    """
    n = np.asarray(n)
    x = np.asarray(x)
    s = np.asarray(s)

    k = len(x)
    df = n - 1
    ssq = s**2
    vsq = df * ssq / n
    v = np.sqrt(vsq)
    sn = np.sum(n)

    # MLES
    tau0 = np.sum(n * vsq / x**2) / sn
    iteration = 1
    while True:
        uh = (-x + np.sqrt(x**2 + 4.0 * tau0 * (vsq + x**2))) / (2.0 * tau0)
        tau = np.sum(n * (vsq + (x - uh) ** 2) / uh**2) / sn
        if abs(tau - tau0) <= 1.0e-7 or iteration > 30:
            break
        iteration += 1
        tau0 = tau

    tauh = np.sqrt(tau)

    elf = 0.0
    clf = 0.0
    for j in range(k):
        clf = (
            clf
            - n[j] * np.log(tauh * uh[j])
            - (n[j] * (vsq[j] + (x[j] - uh[j]) ** 2)) / (2.0 * tauh**2 * uh[j] ** 2)
        )
        elf = elf - n[j] * np.log(v[j]) - n[j] / 2.0

    stat = 2.0 * (elf - clf)
    return np.concatenate([uh, [tauh, stat]])


def mslr_test(sample1, sample2, nr=1000):
    """
    Modified signed-likelihood ratio test (SLRT) for equality of CVs

    Adapted from: https://github.com/benmarwick/cvequality/blob/master/R/functions.R
    """
    k = 2

    n = np.array([len(sample1), len(sample2)])
    x = np.array([bn.nanmean(sample1), bn.nanmean(sample2)])
    s = np.array([bn.nanstd(sample1), bn.nanstd(sample2)])

    gv = np.zeros(nr)
    df = n - 1
    xst0 = _LRT_STAT(n, x, s)
    uh0 = xst0[:k]
    tauh0 = xst0[k]
    stat0 = xst0[k + 1]
    sh0 = tauh0 * uh0
    se0 = tauh0 * uh0 / np.sqrt(n)

    # PB estimates of the mean and SD of the LRT
    for ii in range(nr):
        z = np.random.normal(size=k)
        x_sim = uh0 + z * se0
        ch = np.random.chisquare(df)
        s_sim = sh0 * np.sqrt(ch / df)
        xst = _LRT_STAT(n, x_sim, s_sim)
        gv[ii] = xst[k + 1]

    am = np.mean(gv)
    sd = np.std(gv, ddof=1)
    # end PB estimates

    statm = np.sqrt(2.0 * (k - 1)) * (stat0 - am) / sd + (k - 1)
    pval = 1.0 - stats.chi2.cdf(statm, k - 1)

    return statm, pval


def _annotate_significance(
    df,
    conditions_to_plot,
    column,
    boxplot,
    significance_pairs,
    event_index,
    plot_type="boxplot",
    test="Mann-Whitney",
):
    if significance_pairs is None:
        pairs = list(combinations(df["Condition"].unique(), 2))
    else:
        pairs = significance_pairs
    annotator = Annotator(
        ax=boxplot,
        pairs=pairs,
        data=df[df["Order"] == event_index],
        x="Condition",
        order=conditions_to_plot,
        y=column,
        plot=plot_type,
    )

    if test in STATANNOTATIONS_TESTS:
        if test != "Mann-Whitney":
            annotator.configure(
                test=test,
                text_format="simple",
                loc="inside",
                verbose=False,
                test_short_name=test.capitalize(),
            )
        else:
            annotator.configure(
                test=test, text_format="star", loc="inside", verbose=False
            )
    else:
        if test == "Feltz-Miller":
            custom_long_name = "Feltz-Miller Asymptotic Test"
            custom_short_name = "Feltz-Miller"
            custom_func = feltz_miller_asymptotic_cv_test
            custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

            annotator.configure(
                test=custom_test,
                text_format="simple",
                loc="inside",
                verbose=False,
            )

        elif test == "MSLR":
            custom_long_name = "Modified Signed Likelihood Ratio Test"
            custom_short_name = "MSLR"
            custom_func = mslr_test
            custom_test = StatTest(custom_func, custom_long_name, custom_short_name)

            annotator.configure(
                test=custom_test,
                text_format="simple",
                loc="inside",
                verbose=False,
            )
        else:
            raise ValueError(
                f"Test {test} is not supported. Please use one of the following: {STATANNOTATIONS_TESTS + custom_test}"
            )
    annotator.apply_and_annotate()


def _plot_violinplot(
    df,
    conditions_to_plot,
    column,
    color_palette,
    ax,
    titles,
    share_y_axis,
    plot_significance,
    significance_pairs,
    test="Mann-Whitney",
    hide_outliers=True,
):
    y_min, y_max = [], []
    for event_index in range(df["Order"].nunique()):
        if share_y_axis:
            if event_index > 0:
                ax[event_index].tick_params(
                    axis="y", which="both", left=False, labelleft=False
                )

        if isinstance(ax, np.ndarray):
            current_ax = ax[event_index]
        else:
            current_ax = ax

        violinplot = sns.violinplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            cut=0,
            inner="box",
            ax=current_ax,
            linewidth=2,
            legend="full",
        )

        if hide_outliers:
            data = df[df["Order"] == event_index]
            for condition in conditions_to_plot:
                condition_data = data[data["Condition"] == condition]
                mean = condition_data[column].mean()
                std = condition_data[column].std()
                outliers = condition_data[
                    (condition_data[column] < mean - 3 * std)
                    | (condition_data[column] > mean + 3 * std)
                ]
                # set outliers to NaN
                df.loc[
                    (df["Order"] == event_index)
                    & (df["Condition"] == condition)
                    & (df[column].isin(outliers[column])),
                    column,
                ] = np.nan

        sns.stripplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            order=conditions_to_plot,
            y=column,
            ax=current_ax,
            alpha=0.5,
            color="black",
            dodge=True,
        )

        current_ax.set_xlabel("")
        if event_index > 0:
            current_ax.set_ylabel("")

        if titles is not None:
            current_ax.set_title(titles[event_index])

        current_ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            _annotate_significance(
                df,
                conditions_to_plot,
                column,
                violinplot,
                significance_pairs,
                event_index,
                plot_type="violinplot",
                test=test,
            )

        min_y, max_y = current_ax.get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _plot_boxplot(
    df,
    conditions_to_plot,
    column,
    color_palette,
    ax,
    titles,
    share_y_axis,
    plot_significance,
    significance_pairs,
    hide_outliers=True,
    test="Mann-Whitney",
):
    y_min, y_max = [], []
    for event_index in range(df["Order"].nunique()):
        if share_y_axis:
            if event_index > 0:
                ax[event_index].tick_params(
                    axis="y", which="both", left=False, labelleft=False
                )

        if isinstance(ax, np.ndarray):
            current_ax = ax[event_index]
        else:
            current_ax = ax

        boxplot = sns.boxplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            y=column,
            order=conditions_to_plot,
            hue_order=conditions_to_plot,
            hue="Condition",
            palette=color_palette,
            showfliers=False,
            ax=current_ax,
            dodge=False,
            linewidth=2,
            legend="full",
        )

        if hide_outliers:
            data = df[df["Order"] == event_index]
            for condition in conditions_to_plot:
                condition_data = data[data["Condition"] == condition]
                mean = condition_data[column].mean()
                std = condition_data[column].std()
                outliers = condition_data[
                    (condition_data[column] < mean - 3 * std)
                    | (condition_data[column] > mean + 3 * std)
                ]
                # set outliers to NaN
                df.loc[
                    (df["Order"] == event_index)
                    & (df["Condition"] == condition)
                    & (df[column].isin(outliers[column])),
                    column,
                ] = np.nan

        sns.stripplot(
            data=df[df["Order"] == event_index],
            x="Condition",
            order=conditions_to_plot,
            y=column,
            ax=current_ax,
            alpha=0.5,
            color="black",
            dodge=True,
        )

        current_ax.set_xlabel("")
        # Hide y-axis labels and ticks for all subplots except the first one
        if event_index > 0:
            current_ax.set_ylabel("")

        if titles is not None:
            current_ax.set_title(titles[event_index])

        # remove ticks
        current_ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

        if plot_significance:
            _annotate_significance(
                df,
                conditions_to_plot,
                column,
                boxplot,
                significance_pairs,
                event_index,
                test=test,
            )

        min_y, max_y = current_ax.get_ylim()
        y_min.append(min_y)
        y_max.append(max_y)

    return y_min, y_max


def _set_all_y_limits(ax, y_min, y_max):
    global_min = min(y_min)
    global_max = max(y_max)
    range_padding = (global_max - global_min) * 0.05  # 5% padding
    global_min = global_min - range_padding
    global_max = global_max + range_padding
    for i in range(len(ax)):
        ax[i].set_ylim(global_min, global_max)


def _set_labels_and_legend(
    ax,
    fig,
    conditions_struct,
    conditions_to_plot,
    column,
    y_axis_label,
    legend,
):
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    # Set y label for the first plot
    if y_axis_label is not None:
        ax[0].set_ylabel(y_axis_label)
    else:
        ax[0].set_ylabel(column)

    # Add legend to the right of the subplots
    legend_labels = [
        build_legend(conditions_struct[condition_id], legend)
        for condition_id in conditions_to_plot
    ]

    legend_handles = ax[0].get_legend_handles_labels()[0]

    # Remove the legend from all subplots
    for i in range(len(ax)):
        ax[i].legend_.remove()

    # Place legend to the right of the subplots
    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
        title=None,
        frameon=True,
    )


def violinplot(
    conditions_struct,
    column,
    conditions_to_plot,
    remove_first=False,
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    significance_test: str = "Mann-Whitney",
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
    hide_outliers: bool = True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        number_of_events = data.shape[1]
        if remove_first and number_of_events > 1:
            range_start = 1
        elif not remove_first and number_of_events > 1:
            range_start = 0
        else:
            range_start = 0

        for j in range(range_start, data.shape[1]):
            order = j if not remove_first else j - 1
            for value in data[:, j]:
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": order,
                        column: np.log(value) if log_scale else value,
                    }
                )
    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
    )

    y_min, y_max = _plot_violinplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        hide_outliers=hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
        # set the figure to sharey
        for i in range(len(ax)):
            ax[i].sharey(ax[0])

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig


def boxplot(
    conditions_struct,
    column,
    conditions_to_plot,
    remove_first=False,
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    significance_test: str = "Mann-Whitney",
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
    hide_outliers: bool = True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        number_of_events = data.shape[1]
        if remove_first and number_of_events > 1:
            range_start = 1
        elif not remove_first and number_of_events > 1:
            range_start = 0
        else:
            range_start = 0

        for j in range(range_start, data.shape[1]):
            for value in data[:, j]:
                order = j if not remove_first else j - 1
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": order,
                        column: np.log(value) if log_scale else value,
                    }
                )
    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)
        # set the figure to sharey
        for i in range(len(ax)):
            ax[i].sharey(ax[0])

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig


def violinplot_larval_stage(
    conditions_struct,
    column,
    conditions_to_plot,
    aggregation: str = "mean",
    n_points: int = 100,
    fraction: tuple[float, float] = (0.2, 0.8),
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    significance_test: str = "Mann-Whitney",
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
    hide_outliers: bool = True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    if "rescaled" not in column:
        rescaled_column = column + "_rescaled"
        conditions_struct = rescale_without_flattening(
            conditions_struct, column, rescaled_column, aggregation, n_points
        )
        column = rescaled_column

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[
                :,
                int(fraction[0] * data_of_stage.shape[1]) : int(
                    fraction[1] * data_of_stage.shape[1]
                ),
            ]

            data_of_stage = np.log(data_of_stage) if log_scale else data_of_stage
            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
    )

    y_min, y_max = _plot_violinplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        hide_outliers=hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig


def boxplot_larval_stage(
    conditions_struct,
    column,
    conditions_to_plot,
    aggregation: str = "mean",
    n_points: int = 100,
    fraction: tuple[float, float] = (0.2, 0.8),
    log_scale: bool = True,
    figsize: tuple = None,
    colors=None,
    plot_significance: bool = False,
    significance_pairs=None,
    significance_test: str = "Mann-Whitney",
    legend=None,
    y_axis_label=None,
    titles=None,
    share_y_axis: bool = False,
    hide_outliers: bool = True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    if "rescaled" not in column:
        rescaled_column = column + "_rescaled"
        conditions_struct = rescale_without_flattening(
            conditions_struct, column, rescaled_column, aggregation, n_points
        )
        column = rescaled_column

    # Prepare data
    data_list = []
    for condition_id in conditions_to_plot:
        condition_dict = conditions_struct[condition_id]
        data = condition_dict[column]
        for i in range(data.shape[1]):
            data_of_stage = data[:, i]
            data_of_stage = data_of_stage[
                :,
                int(fraction[0] * data_of_stage.shape[1]) : int(
                    fraction[1] * data_of_stage.shape[1]
                ),
            ]

            data_of_stage = np.log(data_of_stage) if log_scale else data_of_stage
            if aggregation == "mean":
                aggregated_data_of_stage = np.nanmean(data_of_stage, axis=1)
            elif aggregation == "median":
                aggregated_data_of_stage = np.nanmedian(data_of_stage, axis=1)

            for j in range(aggregated_data_of_stage.shape[0]):
                data_list.append(
                    {
                        "Condition": condition_id,
                        "Order": i,
                        column: aggregated_data_of_stage[j],
                    }
                )

    df = pd.DataFrame(data_list)

    fig, ax = _setup_figure(
        df,
        figsize,
        titles,
    )

    y_min, y_max = _plot_boxplot(
        df,
        conditions_to_plot,
        column,
        color_palette,
        ax,
        titles,
        share_y_axis,
        plot_significance,
        significance_pairs,
        hide_outliers,
        test=significance_test,
    )

    _set_labels_and_legend(
        ax,
        fig,
        conditions_struct,
        conditions_to_plot,
        column,
        y_axis_label,
        legend,
    )

    if share_y_axis:
        _set_all_y_limits(ax, y_min, y_max)

    # Make subplots closer together while leaving space for legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    fig = plt.gcf()
    plt.show()

    return fig
