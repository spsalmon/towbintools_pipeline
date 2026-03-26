import argparse
import os
import pickle
import re

import numpy as np
import polars as pl
import scipy
import skimage.filters
from joblib import delayed
from joblib import Parallel
from scipy.signal import find_peaks
from towbintools.foundation.file_handling import write_filemap
from towbintools.foundation.image_handling import normalize_image
from towbintools.foundation.image_handling import read_tiff_file
from whittaker_eilers import WhittakerSmoother


def load_pickles(*pickle_paths):
    loaded_pickles = []
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            files = pickle.load(f)
        loaded_pickles.append(files)
    return loaded_pickles


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filemap", type=str, help="pickled filemap")
    parser.add_argument("-c", "--config", type=str, help="pickled config")
    parser.add_argument("-o", "--output", type=str, help="output")
    parser.add_argument(
        "--input-column", type=str, help="column in filemap to use as input"
    )
    parser.add_argument(
        "--frame-rate", type=float, help="frame rate of the input movies"
    )
    parser.add_argument("--height", type=int, help="minimum height for peak detection")
    parser.add_argument("--width", type=int, help="minimum width for peak detection")
    parser.add_argument(
        "--std-coeff",
        type=float,
        help="coefficient of standard deviation for peak detection prominence",
    )
    parser.add_argument(
        "--dist", type=int, help="minimum distance between peaks in frames"
    )
    return parser.parse_args()


def process_movie(movie_path, std_coeff, distance, width, height, frame_rate):
    movie = read_tiff_file(movie_path)
    pumping_rate = get_pumping_rate(
        movie, std_coeff, distance, width, height, frame_rate
    )
    time_pattern = re.compile(r"Time(\d+)")
    point_pattern = re.compile(r"Point(\d+)")

    time_match = time_pattern.search(movie_path)
    point_match = point_pattern.search(movie_path)

    if time_match and point_match:
        time = int(time_match.group(1))
        point = int(point_match.group(1))
        row = {"Time": time, "Point": point, "pumping_rate": pumping_rate}

        return row
    else:
        raise ValueError("Could not extract time and point from file name.")


def get_pumping_rate(movie, std_coeff, distance, width, height, frame_rate):
    if np.sum(movie) == 0:
        return None

    movie = normalize_image(movie)
    std_fluo = np.std(movie, axis=(1))
    sum_fluo = np.sum(movie, axis=(1))
    sum_fluo = sum_fluo - np.mean(sum_fluo, axis=1, keepdims=True)

    # gaussian filter for each frame
    sum_fluo = [skimage.filters.gaussian(frame, sigma=3) for frame in sum_fluo]
    # sum_fluo = skimage.filters.gaussian(sum_fluo, sigma=3)
    sum_fluo = np.array(sum_fluo)

    middle = sum_fluo.shape[1] // 2
    bulb_positions = []
    pumping_measures = []
    for i in range(sum_fluo.shape[0]):
        peaks, _ = find_peaks(sum_fluo[i], distance=10, height=np.mean(sum_fluo[i]))

        # look for the furthest peaks from the middle
        distances = np.abs(peaks - middle)
        sorted_indices = np.argsort(distances)
        sorted_peaks = peaks[sorted_indices]
        if len(sorted_peaks) < 2:
            return None
        bulb = sorted_peaks[-1]
        pumping_metric = -std_fluo[i][bulb]
        bulb_positions.append(bulb)
        pumping_measures.append(pumping_metric)

    # detrend the pumping measures
    trend = scipy.signal.savgol_filter(pumping_measures, window_length=101, polyorder=5)
    pumping_measures = pumping_measures - trend

    whittaker_smoother = WhittakerSmoother(
        lmbda=0.25, order=2, data_length=len(pumping_measures)
    )

    pumping_measures = whittaker_smoother.smooth(pumping_measures)

    pumping_measures = np.array(pumping_measures) + np.abs(np.min(pumping_measures))

    peaks, _ = find_peaks(
        pumping_measures,
        distance=distance,
        prominence=std_coeff * np.std(pumping_measures),
        width=width,
        height=height,
    )

    return len(peaks) / (len(pumping_measures) / frame_rate)


def main():
    args = get_args()
    filemap = load_pickles(args.filemap)[0]

    output = args.output
    input_column = args.input_column
    frame_rate = args.frame_rate
    height = args.height
    width = args.width
    dist = args.dist
    std_coeff = args.std_coeff

    output_name = os.path.basename(output).split(".")[0]

    input_files = pl.Series(filemap.select(pl.col(input_column))).to_list()

    print(f"Processing {len(input_files)} movies for pumping rate estimation.")

    rows = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_movie)(movie_path, std_coeff, dist, width, height, frame_rate)
        for movie_path in input_files
    )

    df = pl.DataFrame(rows)
    # rename pumping_rate column to include output name
    df = df.rename({"pumping_rate": output_name})

    # sort by Time and Point
    df = df.sort(["Time", "Point"])

    write_filemap(df, output)


if __name__ == "__main__":
    main()
