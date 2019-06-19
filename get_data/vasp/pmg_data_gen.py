import numpy
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import IStructure
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints, VaspInput
from typing import List, Dict, Union


def struct_from_sgnum(sgnum: int,
                      init_coords: Union[List[List[float]], numpy.ndarray] = None,
                      scaling_factor: float = 1,
                      lattice_vectors: Union[List[float], List[List[float]]] = None,
                      species: List[str] = None) -> IStructure:
    if init_coords is None:
        init_coords = [[0, 0, 0], [0, 0, 0.5]]
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species is None:
        species = ["Si"] * len(init_coords)

    lattice = Lattice(
        [x * scaling_factor for vector in lattice_vectors for x in vector]
    )
    structure = IStructure.from_spacegroup(
        sgnum, lattice, species, init_coords
    )
    return structure


def gen_vasp_input(structure: IStructure):
    vasp_input = VaspInput(
        Incar.from_file("INCAR"),
        Kpoints.automatic_gamma_density(structure, 1),
        Poscar(structure),
        Potcar.from_file("POTCAR")
    )
    vasp_input.write_input(".")


if __name__ == "__main__":
    struct = struct_from_sgnum(22)
    gen_vasp_input(struct)
