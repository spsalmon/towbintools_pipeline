import os
import re
import tifffile
import numpy as np
from collections import OrderedDict, defaultdict
from joblib import Parallel, delayed, parallel_config
from aicsimageio.writers import OmeTiffWriter
import ome_types
from ome_types.model import Image, Pixels, Channel

def group_files_by_point(dir_path):
    groups = defaultdict(list)
    for file in os.listdir(dir_path):
        if file.endswith('.tiff') or file.endswith('.tif'):
            try:
                point_number = int(file.split('_')[0])
                groups[point_number].append(file)
            except Exception as e:
                print(f'Could not extract point number from file: {file}, {e}')
    return groups

def process_point(point_list, time, dir_path, fluorescence_pattern, brightfield_pattern, overwrite=False):

    channels_data = []
    channels_metadata = []
    channels_names = []

    for image_path in point_list:
        # check if the image is a fluorescence image
        match = fluorescence_pattern.match(image_path)
        if match:
            point, wavelength = match.groups()
            channel_name = f"Fluorescence_{wavelength}"
        else:
            match = brightfield_pattern.match(image_path)
            if match:
                point = match.group(1)
                channel_name = "BF"
            else:
                continue

        # check if the point has already been processed
        new_filename = f"Time{int(time):05d}_Point{int(point):04d}.tiff"
        output_path = os.path.join(output_dir, new_filename)
        if os.path.exists(output_path) and not overwrite:
            return

        # load the images
        image = tifffile.imread(os.path.join(dir_path, image_path))
        metadata = ome_types.from_tiff(os.path.join(dir_path, image_path))

        channels_data.append(image)
        channels_metadata.append(metadata)
        channels_names.append(channel_name)

    # sort the channels
    sorted_channels = sorted(zip(channels_data, channels_metadata, channels_names), key=lambda x: x[2])
    # put the first element to the end if it is a brightfield image
    if sorted_channels[0][2] == "BF":
        sorted_channels.append(sorted_channels.pop(0))

    # merge the images
    combined_channels_data = [channel[0] for channel in sorted_channels]
    combined_channels_data = np.stack(combined_channels_data, axis=0)

    # merge metadata
    all_metadata = [channel[1].images[0] for channel in sorted_channels]
    print(all_metadata)

    size_x = all_metadata[0].pixels.size_x
    size_y = all_metadata[0].pixels.size_y
    size_c = len(all_metadata)
    size_t = 1
    size_z = all_metadata[0].pixels.size_z

    physical_size_x = all_metadata[0].pixels.physical_size_x
    physical_size_y = all_metadata[0].pixels.physical_size_y

    
    dtype = all_metadata[0].pixels.type
    dimension_order = 'XYCZT'

    id = all_metadata[0].pixels.id

    ordered_channel_names = [channel[2] for channel in sorted_channels]

    combined_channels_metadata = []

    for i, meta in enumerate(all_metadata):
        channel = Channel(name=ordered_channel_names[i], id=f'Channel:{i}', samples_per_pixel=1)
        combined_channels_metadata.append(channel)

    merged_pixels = Pixels(size_x=size_x, size_y=size_y, size_c=size_c, size_t=size_t, size_z=size_z, dimension_order=dimension_order, type=dtype, physical_size_x=physical_size_x, physical_size_y=physical_size_y, id=id, channels=combined_channels_metadata, tiff_data_blocks=[{}])
    merged_metadata = Image(pixels=merged_pixels)
    merged_metadata.id = all_metadata[0].id

    
    merged_metadata.acquisition_date = all_metadata[0].acquisition_date

    merged_OME = ome_types.OME(images = [merged_metadata])

    print(merged_OME.images[0].pixels)

    # Save the image with compression and metadata
    
    combined_channels_data = np.expand_dims(combined_channels_data, axis=(0, 1))

    OmeTiffWriter.save(combined_channels_data, output_path, ome_xml=merged_OME)

def process_directory(dir_path, output_dir, time, overwrite=False):

    print(f"Processing directory: {dir_path}")

    fluorescence_pattern = re.compile(r'(\d+)_\d+_\d+_\d+_Fluorescence_(\d+)_nm_Ex')
    brightfield_pattern = re.compile(r'(\d+)_\d+_\d+_\d+_.*LED.*')

    point_groups = group_files_by_point(dir_path)
    point_lists = list(point_groups.values())

    # Parallel processing
    with parallel_config(backend="loky", n_jobs=-1):
        Parallel()(delayed(process_point)(point_list, time, dir_path, fluorescence_pattern, brightfield_pattern, overwrite) for point_list in point_lists)


def merge_and_rename_images(source_dir, output_dir, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)

    for dir_name in sorted(os.listdir(source_dir))[0]:
        dir_path = os.path.join(source_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name.isdigit():
            process_directory(dir_path, output_dir, dir_name, overwrite)

if __name__ == "__main__":

    experiment_dir = "/mnt/towbin.data/shared/spsalmon/20241018_SQUID_dpy_11_yap1_del"
    source_dir = os.path.join(experiment_dir, "squid_raw")
    output_dir = os.path.join(experiment_dir, "raw")

    overwrite = True

    merge_and_rename_images(source_dir, output_dir, overwrite)