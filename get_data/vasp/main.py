import os
import numpy
from gen_vasp_input import struct_from_sgnum, write_vasp_input
from plot_vasp_output import plot_dos, plot_bs


def gen_bs_from_sgnum(sgnum: int, index: int = None, is_plot: bool = False):

    struct = struct_from_sgnum(
        sgnum=sgnum,
        scaling_factor=4.7,
        lattice_vectors=None,
        init_coords=numpy.random.rand(1, 3),
        species=None,
    )

    write_vasp_input(
        structure=struct,
        kpath_division=20,
        write_dir=".",
    )

    # os.system("mpiexec -n 4 vasp-544-s > vasp.out")

    if index:
        print(index, end=" ")
        pass

    if is_plot:
        plot_dos("vasprun.xml")
        plot_bs("vasprun.xml")


if __name__ == '__main__':
    print("main start")
    gen_bs_from_sgnum(1)
    print("main end")
