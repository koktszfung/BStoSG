import numpy
from pmg_data_gen import frac_coords_from_sgnum
from typing import List
from typing import Union


def dump_poscar(sgnum: int,
                init_coords: Union[List[List[float]], numpy.ndarray] = None,
                scaling_factor=1,
                lattice_vectors: Union[List[float], List[List[float]]] = None,
                species: List[str] = None):
    if init_coords is None:
        init_coords = [[0, 0, 0]]
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    if species is None:
        species = ["Si"] * len(init_coords)

    frac_coords = frac_coords_from_sgnum(
        sgnum, init_coords, scaling_factor, lattice_vectors, species
    )
    with open("POSCAR", "w") as file:
        file.write("frac coords of spacegroup number {}\n".format(sgnum))
        for vector in lattice_vectors:
            file.write("{:.4f} {:.4f} {:.4f}\n".format(*vector))
        file.write("{}\n".format(frac_coords.shape[0]))
        file.write("direct\n")
        for coord in frac_coords:
            file.write("{:.4f} {:.4f} {:.4f}\n".format(*coord))


if __name__ == "__main__":
    dump_poscar(22, numpy.random.rand(1, 3))
