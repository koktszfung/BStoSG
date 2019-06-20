import os
import numpy
from gen_vasp_input import struct_from_sgnum, write_vasp_input
from gen_nn_input import write_nn_input_label_based, write_nn_input_coord_based
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

    write_nn_input_label_based(
        sgnum=sgnum,
        index=index,
        all_hs_path="all_hs_files/all_hslabels.txt",
        vasprun_path="vasprun.xml",
        write_dir="data_upload/label_based/"
    )

    write_nn_input_coord_based(
        structure=struct,
        sgnum=sgnum,
        index=index,
        all_hs_path="all_hs_files/all_hspoints.txt",
        vasprun_path="vasprun.xml",
        write_dir="data_upload/coord_based/"
    )

    if is_plot:
        plot_dos("vasprun.xml")
        plot_bs("vasprun.xml")


if __name__ == '__main__':
    print("main start")
    for i in range(3):
        gen_bs_from_sgnum(
            sgnum=1,
            index=i,
            is_plot=False,
        )
    print("main end")
