import numpy
import pymatgen.core.lattice
import pymatgen.core.structure
from typing import List


def frac_coords_from_sgnum(sgnum: int,
                           scaling_factor=1,
                           lattice_vectors: List[float] = None,
                           species: List[str] = None,
                           init_coords: List[List[float]] = None,
                           ) -> numpy.ndarray:

    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species is None:
        species = ["Si"]
    if init_coords is None:
        init_coords = [[0, 0, 0]]
    lattice = pymatgen.core.lattice.Lattice(
        [x * scaling_factor for x in lattice_vectors]
    )
    structure = pymatgen.core.structure.IStructure.from_spacegroup(
        sgnum, lattice, species, init_coords
    )
    return structure.frac_coords


if __name__ == "__main__":
    print(frac_coords_from_sgnum(1))
