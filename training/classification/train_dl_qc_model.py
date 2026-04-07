# import argparse
import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from towbintools.deep_learning.architectures import PretrainedClassificationModel
from towbintools.deep_learning.utils.augmentation import get_prediction_augmentation
from towbintools.deep_learning.utils.augmentation import get_qc_training_augmentation
from towbintools.deep_learning.utils.dataset import QualityControlDataset
from towbintools.deep_learning.utils.util import create_lightweight_checkpoint

# import shutil

seed = 4
random.seed(seed)

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU

# PyTorch Lightning
pl.seed_everything(seed, workers=True)

image_only = True

dataset_path = "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/datasets/10x_pharynx_qc/pharynx"
model_save_dir = "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/models/10x_pharynx_qc"
project_yaml = os.path.join(dataset_path, "project.yaml")
with open(project_yaml) as f:
    project_config = yaml.safe_load(f)

image_dir = os.path.join(dataset_path, "images")
mask_dir = os.path.join(dataset_path, "masks")
annotation_csv_path = os.path.join(dataset_path, "annotations", "annotations.csv")
annotations_df = pd.read_csv(annotation_csv_path)

database_csv_path = "/mnt/towbin.data/shared/spsalmon/towbinlab_classification_database/datasets/10x_pharynx_qc/pharynx/pharynx_classification_filemap.csv"
database_df = pd.read_csv(database_csv_path)
labels = annotations_df["Class"].values
classes = np.unique(labels).tolist()

train_val_split_ratio = 0.2
test_val_split_ratio = 0.05
annotations_df["Class"] = annotations_df["Class"].replace(
    {"good": "worm", "bad": "error", "unusable": "error"}
)
channels_to_keep = [0]
batch_size = 24
architecture = "efficientnet_b5"

full_normalization_parameters = {
    "type": "percentile",
    "lo": 1,
    "hi": 99,
    "axis": (-2, -1),
}

normalization_type = full_normalization_parameters["type"]
# remove type from normalization_parameters
normalization_parameters = {
    k: v for k, v in full_normalization_parameters.items() if k != "type"
}

# # replace //izbkingston with /mnt
annotations_df["ImagePath"] = annotations_df["ImagePath"].str.replace(
    "//izbkingston", "/mnt"
)
# # replace \ with /
annotations_df["ImagePath"] = annotations_df["ImagePath"].str.replace("\\", "/")
# # add the mask paths to the dataframe
annotations_df["MaskPath"] = annotations_df["ImagePath"].apply(
    lambda x: os.path.join(mask_dir, os.path.basename(x))
)

for idx, row in annotations_df.iterrows():
    image_filename = os.path.basename(row["ImagePath"])
    # find the row in database_df with the same filename where OutputName is equal to image_filename
    matching_row = database_df[database_df["OutputName"] == image_filename]
    # print(f"Matching row for {image_filename}: {matching_row}")
    if not matching_row.empty:
        new_image_path = matching_row.iloc[0]["Image"]
        new_mask_path = matching_row.iloc[0]["Mask"]
        annotations_df.at[idx, "ImagePath"] = new_image_path
        annotations_df.at[idx, "MaskPath"] = new_mask_path

# check that image and mask paths exist
for idx, row in annotations_df.iterrows():
    assert os.path.exists(
        row["ImagePath"]
    ), f"Image path does not exist: {row['ImagePath']}"
    assert os.path.exists(
        row["MaskPath"]
    ), f"Mask path does not exist: {row['MaskPath']}"

train_df, val_test_df = train_test_split(
    annotations_df,
    test_size=train_val_split_ratio + test_val_split_ratio,
    random_state=seed,
    stratify=annotations_df["Class"],
)
val_df, test_df = train_test_split(
    val_test_df,
    test_size=test_val_split_ratio / (test_val_split_ratio + train_val_split_ratio),
    random_state=seed,
    stratify=val_test_df["Class"],
)

train_dataset = QualityControlDataset(
    image_paths=train_df["ImagePath"].tolist(),
    mask_paths=train_df["MaskPath"].tolist() if not image_only else [],
    labels=train_df["Class"].tolist(),
    channels=channels_to_keep,
    classes=classes,
    transform=get_qc_training_augmentation(
        normalization_type, **normalization_parameters
    ),
    resize_method="pad",
)

val_dataset = QualityControlDataset(
    image_paths=val_df["ImagePath"].tolist(),
    mask_paths=val_df["MaskPath"].tolist() if not image_only else [],
    labels=val_df["Class"].tolist(),
    channels=channels_to_keep,
    classes=classes,
    transform=get_prediction_augmentation(
        normalization_type, **normalization_parameters
    ),
    resize_method="pad",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    collate_fn=val_dataset.collate_fn,
)

model = PretrainedClassificationModel(
    architecture,
    len(channels_to_keep) + 1 if not image_only else len(channels_to_keep),
    classes,
    1e-4,
    full_normalization_parameters,
)


# def bn_to_gn(module, groups=32):
#     for name, child in module.named_children():
#         if isinstance(child, nn.BatchNorm2d):
#             C = child.num_features
#             g = min(groups, C)
#             while C % g != 0:
#                 g -= 1
#             setattr(module, name, nn.GroupNorm(g, C, eps=child.eps, affine=True))
#         else:
#             bn_to_gn(child, groups)
#     return module

# model = bn_to_gn(model, groups=32)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for param in m.parameters():
            param.requires_grad = False


model.apply(freeze_bn)

checkpoint_callback = callbacks.ModelCheckpoint(
    dirpath=model_save_dir, save_top_k=1, monitor="val_loss"
)
swa_callback = callbacks.StochasticWeightAveraging(swa_lrs=1e-5)

# configure logger
logger = pl_loggers.TensorBoardLogger(model_save_dir)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    strategy="auto",
    # callbacks=[checkpoint_callback, swa_callback],
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=6,
    gradient_clip_val=0.5,
    detect_anomaly=False,
    deterministic=False,
    logger=logger,
    log_every_n_steps=5,
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(checkpoint_callback.best_model_path)

create_lightweight_checkpoint(
    input_path=checkpoint_callback.best_model_path,
    output_path=os.path.join(model_save_dir, "best_light.ckpt"),
)
