import os
import re
import shutil

import pandas as pd
from joblib import delayed
from joblib import Parallel
from towbintools.foundation.file_handling import get_dir_filemap
from tqdm import tqdm

dir_list = [
    "/mnt/towbin.data/shared/spsalmon/ZIVA_40x_397_405_yap_gfp/20250904_170830_573",
    "/mnt/towbin.data/shared/spsalmon/ZIVA_40x_397_405_yap_gfp/20250905_165257_220",
]

output_dir = "/mnt/towbin.data/shared/spsalmon/ZIVA_40x_397_405_yap_gfp/raw"
backup_dir = "/mnt/towbin.data/shared/spsalmon/ZIVA_40x_397_405_yap_gfp/"

os.makedirs(backup_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def replace_time_in_filename(filename, new_time):
    match = re.search(r"Time(\d+)", filename)

    if not match:
        return ""

    original_time_str = match.group(1)
    original_digit_count = len(original_time_str)

    new_time_str = f"{new_time:0{original_digit_count}d}"

    new_filename = filename.replace(f"Time{original_time_str}", f"Time{new_time_str}")

    return new_filename


def concat_timelapse_experiments(first_filemap, second_filemap):
    # adjust time in the second filemap
    first_filemap["NewTime"] = first_filemap["Time"]
    last_time = first_filemap["Time"].max()
    second_filemap["NewTime"] = second_filemap["Time"]
    second_filemap["NewTime"] += last_time + 1

    # remove rows with no ImagePath
    first_filemap = first_filemap[first_filemap["ImagePath"].notna()]
    second_filemap = second_filemap[second_filemap["ImagePath"].notna()]

    print(second_filemap["NewTime"].values[0], second_filemap["ImagePath"].values[0])

    # change target filename to reflect new time
    first_filemap["NewFilename"] = first_filemap.apply(
        lambda row: os.path.basename(row["ImagePath"]), axis=1
    )
    second_filemap["NewFilename"] = second_filemap.apply(
        lambda row: os.path.basename(
            replace_time_in_filename(row["ImagePath"], row["NewTime"])
        ),
        axis=1,
    )

    combined_filemap = pd.concat([first_filemap, second_filemap], ignore_index=True)

    # Reset index
    combined_filemap.reset_index(drop=True, inplace=True)

    return combined_filemap


def combine_all_and_move(dir_list, output_dir):
    filemaps = [get_dir_filemap(d) for d in dir_list]

    combined_filemap = filemaps[0]
    for next_filemap in filemaps[1:]:
        combined_filemap = concat_timelapse_experiments(combined_filemap, next_filemap)

    combined_filemap["OutputPath"] = combined_filemap.apply(
        lambda row: os.path.join(output_dir, row["NewFilename"]), axis=1
    )

    # sort based on NewTime then Point
    combined_filemap.sort_values(by=["NewTime", "Point"], inplace=True)

    combined_filemap.to_csv(
        os.path.join(backup_dir, "combined_filemap.csv"), index=False
    )

    def move_file(src, dest):
        """Move a single file from source to destination"""
        try:
            shutil.move(src, dest)
            return f"✓ Moved: {os.path.basename(src)}"
        except Exception as e:
            return f"✗ Failed to move {os.path.basename(src)}: {str(e)}"

    # move images in parallel from ImagePath to OutputPath
    Parallel(n_jobs=32, backend="threading")(
        delayed(move_file)(src, dest)
        for src, dest in tqdm(
            zip(combined_filemap["ImagePath"], combined_filemap["OutputPath"]),
            total=len(combined_filemap),
            desc="Moving files",
        )
    )

    return combined_filemap


filemaps = [get_dir_filemap(d) for d in dir_list]

combined_filemap = combine_all_and_move(dir_list, output_dir)
