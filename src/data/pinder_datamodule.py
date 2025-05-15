import os
from typing import Any, Dict, Optional

import pandas as pd
import rootutils
from lightning import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.pinder_dataset import PinderDataset


class PINDERDataModule(LightningDataModule):
    """`LightningDataModule` for the PINDER dataset."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        predicted_structures: bool = False,
        high_quality: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the `PINDERDataModule`.

        Args:
            data_dir: Data for pinder. Defaults to "data/processed".
            predicted_structures: Whether to use predicted structures. Defaults to True.
            batch_size: Batch size. Defaults to 64.
            num_workers: Number of workers for parallel processing. Defaults to 0.
            pin_memory: Whether to pin memory. Defaults to True.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # get metadata
        metadata = pd.read_csv(os.path.join(self.hparams.data_dir, "metadata.csv"))

        def get_files(split: str, complex_types: list) -> list:
            file_df = metadata[
                (metadata["split"] == split) & (metadata["complex"].isin(complex_types))
            ]
            file_df["file_paths"] = file_df.apply(
                lambda row: os.path.join(
                    "./data/processed", row["complex"], row["split"], row["file_paths"]
                ),
                axis=1,
            )
            return file_df["file_paths"].tolist()

        complex_types = ["apo", "predicted"] if self.hparams.predicted_structures else ["apo"]
        self.train_files = get_files("train", complex_types)
        self.val_files = get_files("val", complex_types)
        self.test_files = get_files("test", complex_types)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = PinderDataset(self.train_files)
            self.data_val = PinderDataset(self.val_files)
            self.data_test = PinderDataset(self.test_files)

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    datamodule = PINDERDataModule()
    datamodule.setup()
    # print(datamodule.train_files[64])
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    i = 0
    for data in train_loader:
        i+=1
        if i ==64:
            print(data)
        if i ==65:
            print(data)
        if i==63:
            print(data)
