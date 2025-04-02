import argparse
import os
import shutil

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as pl_loggers
import torch.nn as nn
import yaml
from towbintools.deep_learning.deep_learning_tools import (
    create_pretrained_segmentation_model,
)
from towbintools.deep_learning.deep_learning_tools import create_segmentation_model
from towbintools.deep_learning.utils.augmentation import get_prediction_augmentation
from towbintools.deep_learning.utils.augmentation import get_training_augmentation
from towbintools.deep_learning.utils.dataset import create_segmentation_dataloaders
from towbintools.deep_learning.utils.dataset import (
    create_segmentation_dataloaders_from_filemap,
)
from towbintools.deep_learning.utils.dataset import (
    create_segmentation_training_dataframes_and_dataloaders,
)
from towbintools.deep_learning.utils.loss import BCELossWithIgnore
from towbintools.deep_learning.utils.loss import FocalTverskyLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", required=True)
    args = parser.parse_args()
    return args


config_file = get_args().config
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

image_directories = config.get("image_directories", None)
mask_directories = config.get("mask_directories", None)
training_filemap = config.get("training_filemap", None)
training_dataframes = config.get("training_dataframes", None)
validation_dataframes = config.get("validation_dataframes", None)
test_dataframes = config.get("test_dataframes", None)
save_dir = config["save_dir"]
model_name = config["model_name"]
pretrained = config.get("pretrained", True)
n_classes = config.get("n_classes", 1)
channels_to_segment = config.get("channels_to_segment", [0])
architecture = config.get("architecture", "UnetPlusPlus")
pretrained_encoder = config.get("pretrained_encoder", "efficientnet-b4")
pretrained_weights = config.get("pretrained_weights", "image-micronet")
deep_supervision = config.get("deep_supervision", False)
learning_rate = config.get("learning_rate", 1e-4)
loss = config.get("loss", "FocalTversky")
value_to_ignore = config.get("value_to_ignore", None)

if loss == "FocalTversky":
    criterion = FocalTverskyLoss(ignore_index=value_to_ignore)
elif loss == "BCE":
    criterion = BCELossWithIgnore(ignore_index=value_to_ignore)
elif loss == "CrossEntropy":
    criterion = nn.CrossEntropyLoss(ignore_index=value_to_ignore)
else:
    raise ValueError(f"{loss} loss not implemented yet")

full_normalization_parameters = config.get(
    "normalization_parameters",
    {"type": "percentile", "lo": 1, "hi": 99, "axis": (-2, -1)},
)

normalization_type = full_normalization_parameters["type"]
# remove type from normalization_parameters
normalization_parameters = {
    k: v for k, v in full_normalization_parameters.items() if k != "type"
}

train_on_tiles = config.get("train_on_tiles", True)
tiler_params = config.get(
    "tiler_params", {"tile_size": [512, 512], "tile_step": [256, 256]}
)
max_epochs = config.get("max_epochs", 100)
batch_size = config.get("batch_size", 5)
accumulate_grad_batches = config.get("accumulate_grad_batches", 8)
num_workers = config.get("num_workers", 8)
save_best_k_models = config.get("save_best_k_models", 2)
train_val_split_ratio = config.get("train_val_split_ratio", 0.25)
train_test_split_ratio = config.get("train_test_split_ratio", 0.1)
checkpoint_path = config.get("continue_training_from_checkpoint", None)

model_save_dir = os.path.join(save_dir, model_name)
os.makedirs(model_save_dir, exist_ok=True)

# copy the config file to the model save directory
shutil.copy(config_file, model_save_dir)

input_channels = len(channels_to_segment)

if training_filemap is not None:
    _, _, train_loader, val_loader = create_segmentation_dataloaders_from_filemap(
        training_filemap,
        save_dir=model_save_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        channels=channels_to_segment,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=get_training_augmentation(
            normalization_type, **normalization_parameters
        ),
        validation_transform=get_prediction_augmentation(
            normalization_type, **normalization_parameters
        ),
        validation_set_ratio=train_val_split_ratio,
        test_set_ratio=train_test_split_ratio,
    )

elif image_directories is not None and mask_directories is not None:
    # create dataframes and dataloaders
    (
        training_dataframe,
        validation_dataframe,
        train_loader,
        val_loader,
    ) = create_segmentation_training_dataframes_and_dataloaders(
        image_directories,
        mask_directories,
        save_dir=model_save_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        train_on_tiles=train_on_tiles,
        channels=channels_to_segment,
        tiler_params=tiler_params,
        training_transform=get_training_augmentation(
            normalization_type, **normalization_parameters
        ),
        validation_transform=get_prediction_augmentation(
            normalization_type, **normalization_parameters
        ),
        validation_set_ratio=train_val_split_ratio,
        test_set_ratio=train_test_split_ratio,
    )

elif training_dataframes is not None and validation_dataframes is not None:
    # combine the training dataframes together
    training_df = []
    for training_dataframe in training_dataframes:
        training_dataframe = pd.read_csv(training_dataframe)
        training_df.append(training_dataframe)
    validation_df = []
    for validation_dataframe in validation_dataframes:
        validation_dataframe = pd.read_csv(validation_dataframe)
        validation_df.append(validation_dataframe)

    combined_training_dataframe = pd.concat(training_df, ignore_index=True)
    combined_validation_dataframe = pd.concat(validation_df, ignore_index=True)

    os.makedirs(os.path.join(model_save_dir, "dataframe_backup"), exist_ok=True)
    # backup the dataframes
    combined_training_dataframe.to_csv(
        os.path.join(model_save_dir, "dataframe_backup", "training_dataframe.csv"),
        index=False,
    )
    combined_validation_dataframe.to_csv(
        os.path.join(model_save_dir, "dataframe_backup", "validation_dataframe.csv"),
        index=False,
    )

    if test_dataframes is not None:
        # combine the test dataframes together
        test_df = []
        for test_dataframe in test_dataframes:
            test_dataframe = pd.read_csv(test_dataframe)
            test_df.append(test_dataframe)

        combined_test_dataframe = pd.concat(test_df, ignore_index=True)
        combined_test_dataframe.to_csv(
            os.path.join(model_save_dir, "dataframe_backup", "test_dataframe.csv"),
            index=False,
        )

    # create dataloaders
    train_loader, val_loader = create_segmentation_dataloaders(
        combined_training_dataframe,
        combined_validation_dataframe,
        batch_size=batch_size,
        num_workers=num_workers,
        channels=channels_to_segment,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=get_training_augmentation(
            normalization_type, **normalization_parameters
        ),
        validation_transform=get_prediction_augmentation(
            normalization_type, **normalization_parameters
        ),
    )

# initialize model
if pretrained:
    model = create_pretrained_segmentation_model(
        input_channels=input_channels,
        n_classes=n_classes,
        architecture=architecture,
        encoder=pretrained_encoder,
        pretrained_weights=pretrained_weights,
        normalization=full_normalization_parameters,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        criterion=criterion,
    )
else:
    model = create_segmentation_model(
        n_classes=n_classes,
        input_channels=input_channels,
        architecture=architecture,
        normalization=full_normalization_parameters,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
        deep_supervision=deep_supervision,
        criterion=criterion,
    )

checkpoint_callback = callbacks.ModelCheckpoint(
    dirpath=model_save_dir, save_top_k=save_best_k_models, monitor="val_loss"
)
swa_callback = callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

# configure logger
logger = pl_loggers.TensorBoardLogger(model_save_dir)

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu",
    strategy="auto",
    callbacks=[checkpoint_callback, swa_callback],
    accumulate_grad_batches=accumulate_grad_batches,
    gradient_clip_val=0.5,
    detect_anomaly=False,
    logger=logger,
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(checkpoint_callback.best_model_path)
