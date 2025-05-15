from typing import List

import __main__
import rootutils
import torch
from torch_geometric.data import Dataset
import os
import pandas as pd

# setup root dir and pythonpath
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.prepare_data import CropPairedPDB

setattr(__main__, "CropPairedPDB", CropPairedPDB)


class PinderDataset(Dataset):
    """Pinder dataset.

    Args:
        Dataset: PyTorch Geometric Dataset.
    """

    def __init__(self, file_paths: List[str]) -> None:
        """Initialize the PinderDataset.

        Args:
            file_paths: List of file paths.
        """
        super().__init__()
        self.file_paths = file_paths

    @property
    def processed_file_names(self) -> List[str]:
        """Return the processed file names.

        Returns:
            List[str]: List of processed
        """
        return self.file_paths

    def len(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.processed_file_names)

    def get(self, idx) -> CropPairedPDB:
        """Get the data at the given index.

        Args:
            idx: Index of the data.

        Returns:
            CropPairedPDB: CropPairedPDB object.
        """
        data = torch.load(self.processed_file_names[idx], weights_only=False)
        return data


if __name__ == "__main__":
    # file_paths = ["/home/sukanya/pinder_challenge/pinder_challenge-1/data/processed/apo/test/1a19__A1_P11540--1a19__B1_P11540.pt"]
    # dataset = PinderDataset(file_paths=file_paths)
    # print(dataset[0]['ligand', 'ligand'].edge_index)
    
    test = os.listdir("./data/processed/apo/test")
    train = os.listdir("./data/processed/apo/train")
    val = os.listdir("./data/processed/apo/val")
    
    data = {"split":[], "complex":[],"file_paths":[]}
    for te in test:
        graphdata = torch.load("./data/processed/apo/test/"+te)
        if graphdata["receptor"].x.shape[0] == graphdata["receptor"].pos.shape[0]:
            if graphdata["ligand"].x.shape[0] == graphdata["ligand"].pos.shape[0]:
                if graphdata["receptor"].y.shape == graphdata["receptor"].pos.shape:
                    if graphdata["ligand"].y.shape == graphdata["ligand"].pos.shape:
                        if graphdata["receptor"].x.shape[0] !=1:
                            if graphdata["ligand"].x.shape[0] !=1:
                                data["split"].append("test")
                                data["complex"].append("apo")
                                data["file_paths"].append(te)
                
    for te in val:
        graphdata = torch.load("./data/processed/apo/val/"+te)
        if graphdata["receptor"].x.shape[0] == graphdata["receptor"].pos.shape[0]:
            if graphdata["ligand"].x.shape[0] == graphdata["ligand"].pos.shape[0]:
                if graphdata["receptor"].y.shape == graphdata["receptor"].pos.shape:
                    if graphdata["ligand"].y.shape == graphdata["ligand"].pos.shape:
                        if graphdata["receptor"].x.shape[0] !=1:
                            if graphdata["ligand"].x.shape[0] !=1:
                                data["split"].append("val")
                                data["complex"].append("apo")
                                data["file_paths"].append(te)
                
    for te in train:
        graphdata = torch.load("./data/processed/apo/train/"+te)
        if graphdata["receptor"].x.shape[0] == graphdata["receptor"].pos.shape[0]:
            if graphdata["ligand"].x.shape[0] == graphdata["ligand"].pos.shape[0]:
                if graphdata["receptor"].y.shape == graphdata["receptor"].pos.shape:
                    if graphdata["ligand"].y.shape == graphdata["ligand"].pos.shape:
                        if graphdata["receptor"].x.shape[0] !=1:
                            if graphdata["ligand"].x.shape[0] !=1:
                                data["split"].append("train")
                                data["complex"].append("apo")
                                data["file_paths"].append(te)
    
    train = os.listdir("./data/processed/predicted/train")
                
    for te in train:
        graphdata = torch.load("./data/processed/predicted/train/"+te)
        if graphdata["receptor"].x.shape[0] == graphdata["receptor"].pos.shape[0]:
            if graphdata["ligand"].x.shape[0] == graphdata["ligand"].pos.shape[0]:
                if graphdata["receptor"].y.shape == graphdata["receptor"].pos.shape:
                    if graphdata["ligand"].y.shape == graphdata["ligand"].pos.shape:
                        if graphdata["receptor"].x.shape[0] !=1:
                            if graphdata["ligand"].x.shape[0] !=1:
                                data["split"].append("train")
                                data["complex"].append("predicted")
                                data["file_paths"].append(te)

    pd.DataFrame(data).to_csv("./data/processed/metadata.csv", index=False)