from pmg_data_gen import frac_coords_from_sgnum
from typing import List


def dump_poscar(sgnum,
                scaling_factor=1,
                lattice_vectors: List[float] = None):
    if lattice_vectors is None:
        lattice_vectors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    frac_coords = frac_coords_from_sgnum(
        sgnum, scaling_factor, lattice_vectors
    )
    with open("POSCAR", "w") as file:
        file.write("frac coords of spacegroup number {}\n".format(sgnum))
        for vector in lattice_vectors:
            file.write("{} {} {}\n".format(*vector))
        file.write("{}\n".format(frac_coords.shape[0]))
        file.write("direct\n")
        for coord in frac_coords:
            file.write("{} {} {}\n".format(*coord))


if __name__ == "__main__":
    dump_poscar(1)
