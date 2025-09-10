import os
import re
import shutil

from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

experiment_dir = (
    "/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/"
)
image_dir = os.path.join(experiment_dir, "raw")
channels_to_output = {
    "GFP,mCherry,GFP,mCherry": "raw",
    "GFP,mCherry": "raw_zstack",
}

for subdir in channels_to_output.values():
    os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)


def move_image(filename, experiment_dir):
    match = re.search(r"Channel([A-Za-z0-9_,]+)_Seq", filename)

    if not match:
        print(f"Filename {filename} does not match expected pattern, skipping.")
        return

    channel_str = match.group(1)

    if channel_str not in channels_to_output:
        print(f"Channel combination {channel_str} not in channels_to_output, skipping.")
        return

    output_subdir = channels_to_output[channel_str]
    output_dir = os.path.join(experiment_dir, output_subdir)
    output_filename = os.path.basename(filename)
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping move.")
        return

    shutil.move(filename, output_path)


Parallel(n_jobs=32, prefer="threads")(
    delayed(move_image)(filename, experiment_dir)
    for filename in tqdm([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
)
