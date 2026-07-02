import pickle

import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from towbintools.deep_learning.architectures import KeypointDetection1DModel
from towbintools.deep_learning.utils.dataset import KeypointDetection1DTrainingDataset

if __name__ == "__main__":
    X = pickle.load(open("X_molt_detection.pickle", "rb"))
    y = pickle.load(open("y_gaussian_molt_detection.pickle", "rb"))
    keypoints_all = pickle.load(open("keypoints_molt_detection.pickle", "rb"))

    y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
    X_raw_train, X_raw_test, keypoints_train, keypoints_test = train_test_split(
        X, keypoints_all, test_size=0.2, random_state=42
    )

    batch_size = 64

    model = KeypointDetection1DModel(
        input_channels=2,
        n_classes=4,
        learning_rate=1e-4,
        activation="sigmoid",
    )

    train_dataset = KeypointDetection1DTrainingDataset(
        inputs=X_raw_train,
        heatmap_targets=y_train,
        index_targets=keypoints_train,
        enforce_divisibility_by=32,
        resize_method="pad",
    )

    val_dataset = KeypointDetection1DTrainingDataset(
        inputs=X_raw_test,
        heatmap_targets=y_test,
        index_targets=keypoints_test,
        enforce_divisibility_by=32,
        resize_method="pad",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=val_dataset.collate_fn,
    )

    from pytorch_lightning import callbacks

    model_save_dir = "./model_checkpoints"
    save_best_k_models = 1
    max_epochs = 100
    accumulate_grad_batches = 1

    device = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=model_save_dir, save_top_k=save_best_k_models, monitor="val_loss"
    )
    swa_callback = callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        strategy="auto",
        callbacks=[checkpoint_callback, swa_callback],
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=0.5,
        detect_anomaly=False,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
