import os
import re
import shutil

from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

experiment_dir = "/mnt/towbin.data/shared/nschoonjans/20260402_Ziva_60X_397-405-AID_nuclear_shape_chambers"
image_dir = os.path.join(experiment_dir, "raw")

re_template = r"Channel\s*([A-Za-z0-9_,\s]+?)\s*_Seq"
pattern_to_output = {
    "WF BF,WF mCherry,WF GFP": "raw",
    "SD mCherry": "raw_stacks",
}

# re_template = r"\.(\w+)$"
# pattern_to_output = {
#     "tiff": "raw",
#     "nd2": "raw_movies",
# }

for subdir in pattern_to_output.values():
    os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)


def move_image(filename, experiment_dir, re_template, pattern_to_output):
    match = re.search(re_template, filename)

    if not match:
        print(f"Filename {filename} does not match expected pattern, skipping.")
        return

    pattern = match.group(1)

    if pattern not in pattern_to_output:
        print(f"Pattern {pattern} not in pattern_to_output, skipping.")
        return

    output_subdir = pattern_to_output[pattern]
    output_dir = os.path.join(experiment_dir, output_subdir)
    output_filename = os.path.basename(filename)
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping move.")
        return

    shutil.move(filename, output_path)


Parallel(n_jobs=32, prefer="threads")(
    delayed(move_image)(filename, experiment_dir, re_template, pattern_to_output)
    for filename in tqdm([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
)
