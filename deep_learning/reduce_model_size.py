import os

from towbintools.deep_learning.utils.util import create_lightweight_checkpoint

if __name__ == "__main__":
    model_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_segmentation_database/models/body/SAMannotations_singleclass_UnetPlusPlus_efficientnet-b4_image-micronet_FocalTversky/"
    model_name = "epoch=587-step=16464.ckpt"
    output_name = "best_light.ckpt"

    create_lightweight_checkpoint(
        input_path=os.path.join(model_dir, model_name),
        output_path=os.path.join(model_dir, output_name),
    )
