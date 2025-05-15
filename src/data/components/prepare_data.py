import multiprocessing
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import rootutils
import torch
from loguru import logger
from pinder.core import PinderSystem, get_index
from pinder.core.loader.geodata import PairedPDB, structure2tensor
from pinder.core.loader.structure import Structure
from tqdm.auto import tqdm

# setup root dir and pythonpath
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

try:
    from torch_cluster import knn_graph

    torch_cluster_installed = True
except ImportError:
    logger.warning(
        "torch-cluster is not installed!"
        "Please install the appropriate library for your pytorch installation."
        "See https://github.com/rusty1s/pytorch_cluster/issues/185 for background."
    )
    torch_cluster_installed = False


def create_lr_files(system_id: str, apo_complex_path: str, save_path: str):
    apo_r_path = os.path.join(save_path, f"apo_r_{system_id}.pdb")
    apo_l_path = os.path.join(save_path, f"apo_l_{system_id}.pdb")
    native_path = apo_complex_path.with_name(apo_complex_path.stem + f"{system_id}.pdb")
    with open(native_path) as infile, open(apo_r_path, "w") as output_r, open(
        apo_l_path, "w"
    ) as output_l:

        for line in infile:
            # Check if the line is an ATOM or HETATM line and has a chain ID at position 21
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                if chain_id == "R":
                    output_r.write(line)
                elif chain_id == "L":
                    output_l.write(line)
            else:
                # Write other lines (e.g., HEADER, REMARK) to both files
                output_r.write(line)
                output_l.write(line)
    return apo_r_path, apo_l_path


class CropPairedPDB(PairedPDB):
    @classmethod
    def from_crop_system(
        cls,
        system_id: str,
        root: str = "./data/",
        k: int = 10,
        add_edges: bool = True,
        predicted_structures: bool = True,
        split: str = "train",
    ) -> None:
        system = PinderSystem(system_id)
        # Create directories if they do not exist
        for subdir in ["apo", "holo", "predicted"]:
            os.makedirs(Path(root) / "raw" / subdir / split, exist_ok=True)

        try:
            holo_complex, apo_complex, pred_complex = system.create_masked_bound_unbound_complexes(
                renumber_residues=True
            )
            for complex_type, complex_obj in zip(
                ["apo", "holo", "predicted"], [apo_complex, holo_complex, pred_complex]
            ):
                complex_obj.to_pdb(
                    Path(root) / "raw" / complex_type / split / f"{system_id}_complex.pdb"
                )
        except Exception as e:
            logger.error(f"Error in writing PDB files: {e}, {system_id}")
            return None

        if predicted_structures:
            apo_complex = pred_complex
            save_path = os.path.join(root, "processed", "predicted", split)
        else:
            save_path = os.path.join(root, "processed", "apo", split)

        # create the directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

        graph = cls.from_structure_pair(
            holo_complex=holo_complex,
            apo_complex=apo_complex,
            add_edges=add_edges,
            k=k,
        )
        torch.save(graph, os.path.join(save_path, f"{system_id}.pt"))

    @classmethod
    def from_structure_pair(
        cls,
        holo_complex: Structure,
        apo_complex: Structure,
        add_edges: bool = True,
        k: int = 10,
    ) -> PairedPDB:
        def get_structure_props(structure: Structure, start: int, end: Optional[int]):
            calpha = structure.filter("atom_name", mask=["CA"])
            return structure2tensor(
                atom_coordinates=structure.coords[start:end],
                atom_types=structure.atom_array.atom_name[start:end],
                element_types=structure.atom_array.element[start:end],
                residue_coordinates=calpha.coords[start:end],
                residue_types=calpha.atom_array.res_name[start:end],
                residue_ids=calpha.atom_array.res_id[start:end],
            )

        graph = cls()
        r_h = (holo_complex.dataframe["chain_id"] == "R").sum()
        r_a = (apo_complex.dataframe["chain_id"] == "R").sum()

        holo_r_props = get_structure_props(holo_complex, 0, r_h)
        holo_l_props = get_structure_props(holo_complex, r_h, None)
        apo_r_props = get_structure_props(apo_complex, 0, r_a)
        apo_l_props = get_structure_props(apo_complex, r_a, None)

        graph["ligand"].x = apo_l_props["atom_types"]
        graph["ligand"].pos = apo_l_props["atom_coordinates"]
        graph["receptor"].x = apo_r_props["atom_types"]
        graph["receptor"].pos = apo_r_props["atom_coordinates"]
        graph["ligand"].y = holo_l_props["atom_coordinates"]
        graph["receptor"].y = holo_r_props["atom_coordinates"]

        if add_edges and torch_cluster_installed:
            graph["ligand", "ligand"].edge_index = knn_graph(graph["ligand"].pos, k=k)
            graph["receptor", "receptor"].edge_index = knn_graph(graph["receptor"].pos, k=k)

        return graph


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--predicted_structures", action="store_true")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    predicted_structures = args.predicted_structures

    # get indices for train, validation, and test splits
    indices = get_index()

    if predicted_structures:
        query = '(split == "{split}") and ((apo_R == False and apo_L == False) and (predicted_R==True and predicted_L==True))'
    else:
        query = '(split == "{split}") and (apo_R == True and apo_L == True)'

    system_idx = indices.query(query.format(split=args.split)).reset_index(drop=True)

    system_ids = system_idx.id.tolist()[66000:]

    def process_system_id(system_id: str):
        graph = CropPairedPDB.from_crop_system(
            system_id,
            predicted_structures=predicted_structures,
            k=args.k,
            split=args.split,
        )
        return graph

    with multiprocessing.Pool(args.n_jobs) as pool:
        results = list(tqdm(pool.imap(process_system_id, system_ids), total=len(system_ids)))
