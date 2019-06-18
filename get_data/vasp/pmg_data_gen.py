import numpy
import pymatgen.core.lattice
import pymatgen.core.structure
from typing import List
from typing import Union


def struct_from_sgnum(sgnum: int,
                      init_coords: Union[List[List[float]], numpy.ndarray] = None,
                      scaling_factor: float = 1,
                      lattice_vectors: Union[List[float], List[List[float]]] = None,
                      species: List[str] = None) -> pymatgen.core.structure.IStructure:
    if init_coords is None:
        init_coords = [[0, 0, 0]]
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species is None:
        species = ["Si"] * len(init_coords)

    lattice = pymatgen.core.lattice.Lattice(
        [x * scaling_factor for vector in lattice_vectors for x in vector]
    )
    structure = pymatgen.core.structure.IStructure.from_spacegroup(
        sgnum, lattice, species, init_coords
    )
    return structure


if __name__ == "__main__":
    print(struct_from_sgnum(1).frac_coords)
