import numpy
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import IStructure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, Kpoints, VaspInput
from typing import List, Union


def struct_from_sgnum(sgnum: int,
                      scaling_factor: float = 1,
                      lattice_vectors: Union[List[List[float]], numpy.ndarray] = None,
                      init_coords: Union[List[List[float]], numpy.ndarray] = None,
                      species: List[str] = None) -> IStructure:
    if init_coords is None:
        init_coords = [[0, 0, 0]]
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
    return structure.get_primitive_structure()


def write_vasp_input(structure: IStructure,
                     kpath_division: int,
                     write_dir: str = "."):
    vasp_input = VaspInput(
        Incar.from_file("INCAR"),
        Kpoints.automatic_linemode(kpath_division, HighSymmKpath(structure)),
        Poscar(structure),
        Potcar.from_file("POTCAR")
    )
    vasp_input.write_input(write_dir)


if __name__ == "__main__":
    print("main start")
    struct = struct_from_sgnum(
        sgnum=1,
        scaling_factor=4.7,
        init_coords=numpy.random.rand(1, 3),
    )
    write_vasp_input(
        structure=struct,
        kpath_division=20,
    )
    print("main end")
