import numpy
from pmg_data_gen import struct_from_sgnum
from typing import List, Dict, Union


def dump_poscar(sgnum: int,
                init_coords: Union[List[List[float]], numpy.ndarray] = None,
                scaling_factor: float = 1,
                lattice_vectors: Union[List[float], List[List[float]]] = None,
                species_dict: Dict[str, int] = None):
    if init_coords is None:
        init_coords = [[0, 0, 0]]
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species_dict is None:
        species_dict = {"Si": len(init_coords)}

    species = []
    for key, val in species_dict.items():
        species.extend([key] * int(val))
    structure = struct_from_sgnum(
        sgnum, init_coords, scaling_factor, lattice_vectors, species
    )

    frac_coords = structure.frac_coords
    species_dict = structure.composition
    with open("POSCAR", "w") as file:
        file.write("frac coords of spacegroup number {}\n".format(sgnum))
        file.write("{:.4f}\n".format(scaling_factor))
        for vector in lattice_vectors:
            file.write("{:.4f} {:.4f} {:.4f}\n".format(*vector))
        file.write(("{} "*len(species_dict)).format(*species_dict.keys())+"\n")
        file.write(("{:.0f} "*len(species_dict)).format(*species_dict.values())+"\n")
        file.write("direct\n")
        for coord in frac_coords:
            file.write("{:.4f} {:.4f} {:.4f}\n".format(*coord))


if __name__ == "__main__":
    dump_poscar(
        sgnum=14,
        init_coords=numpy.random.rand(1, 3),
        scaling_factor=4,
        lattice_vectors=None,
        species_dict=None
    )
