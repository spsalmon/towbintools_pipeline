import os

import bioio_nd2
import ome_types
from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter
from joblib import delayed
from joblib import Parallel
from ome_types.model import Image
from ome_types.model import Pixels
from tqdm import tqdm

nd2_dir = "/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw_nd2/"
nd2_files = [
    os.path.join(nd2_dir, f) for f in os.listdir(nd2_dir) if f.endswith(".nd2")
]
output_dir = (
    "/mnt/towbin.data/shared/spsalmon/20250904_170830_573_ZIVA_40x_397_405_yap_gfp/raw/"
)

os.makedirs(output_dir, exist_ok=True)


def convert_nd2_to_tiff(input_path, output_dir):
    output_filename = os.path.basename(input_path).replace(".nd2", ".ome.tiff")
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping conversion.")
        return

    img = BioImage(input_path, reader=bioio_nd2.Reader)
    instrument = img.metadata.instruments
    old_pixels = img.metadata.images[0].pixels

    new_pixels = Pixels(
        size_x=old_pixels.size_x,
        size_y=old_pixels.size_y,
        size_c=old_pixels.size_c,
        size_t=old_pixels.size_t,
        size_z=old_pixels.size_z,
        dimension_order=old_pixels.dimension_order,
        physical_size_x=old_pixels.physical_size_x,
        physical_size_y=old_pixels.physical_size_y,
        id=0,
        type=old_pixels.type,
        channels=old_pixels.channels,
        tiff_data_blocks=[{}],
    )

    image = Image(pixels=new_pixels)
    image.acquisition_date = img.metadata.images[0].acquisition_date
    ome = ome_types.OME(images=[image], instruments=instrument)

    OmeTiffWriter.save(
        img.get_image_data("ZCYX"),
        output_path,
        dim_order="ZCYX",
        ome_xml=ome,
        tifffile_kwargs={
            "compression": "zlib",
            "compressionargs": {"level": 8},
        },
    )


Parallel(n_jobs=32, prefer="processes")(
    delayed(convert_nd2_to_tiff)(f, output_dir) for f in tqdm(nd2_files)
)
