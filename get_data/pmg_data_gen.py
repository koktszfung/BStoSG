import numpy
import pymatgen.core.lattice
import pymatgen.core.structure
from typing import List
from typing import Union


def frac_coords_from_sgnum(sgnum: int,
                           init_coords: Union[List[List[float]], numpy.ndarray] = None,
                           scaling_factor=1,
                           lattice_vectors: Union[List[float], List[List[float]]] = None,
                           species: List[str] = None) -> numpy.ndarray:
    if init_coords is None:
        init_coords = [[0, 0, 0]]
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species is None:
        species = ["Si"] * len(init_coords)

    lattice = pymatgen.core.lattice.Lattice(
        [x * scaling_factor for x in lattice_vectors]
    )
    structure = pymatgen.core.structure.IStructure.from_spacegroup(
        sgnum, lattice, species, init_coords
    )
    return structure.frac_coords


if __name__ == "__main__":
    print(frac_coords_from_sgnum(1))
