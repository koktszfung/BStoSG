import os
import numpy
from gen_vasp_input import struct_from_sgnum, write_vasp_input
from gen_nn_input import write_nn_input_label_based, write_nn_input_coord_based
from plot_vasp_output import plot_dos, plot_bs
import argparse


def vasp(sgnum: int, index: int,):
    try:
        print("\tvasp {} start".format(index))
        os.system("mpiexec -n 4 vasp-544-s > vasp.out")
    except Exception as e:
        print("\tskipped due to {}".format(e))
        return
    else:
        os.system("cp vasprun.xml vaspruns/vasprun_{}_{}".format(sgnum, index))
        print("\tvasp end")


def gen_bs_from_sgnum(division: int,
                      sgnum: int,
                      index: int = None,
                      plot: bool = False):
    struct = struct_from_sgnum(
        sgnum=sgnum,
        scaling_factor=4,
        lattice_vectors=None,
        init_coords=numpy.random.rand(1, 3),
        species=None,
    )

    write_vasp_input(
        structure=struct,
        kpath_division=division,
        write_dir=".",
    )

    vasp(sgnum=sgnum, index=index)

    write_nn_input_label_based(
        sgnum=sgnum,
        index=index,
        all_hs_path="all_hs_files/all_hslabels.txt",
        vasprun_path="vasprun.xml",
        write_dir="input_data/label_based/"
    )

    write_nn_input_coord_based(
        structure=struct,
        sgnum=sgnum,
        index=index,
        all_hs_path="all_hs_files/all_hspoints.txt",
        vasprun_path="vasprun.xml",
        write_dir="input_data/coord_based/"
    )

    if plot:
        # plot_dos("vasprun.xml")
        plot_bs("vasprun.xml")


def kwargs_from_file(file_path):
    with open(file_path) as file:
        return {line.split()[0]: line.split()[1] for line in file}


def main(division: int,
         sgnum: int,
         start: int = 0,
         total: int = 1,
         plot: bool = False):
    print("main {} start".format(sgnum))
    for i in range(start, start + total):
        gen_bs_from_sgnum(
            division=division,
            sgnum=sgnum,
            index=i,
            plot=i == start + total - 1 and plot,
        )
    print("main end")


if __name__ == '__main__':
    main(**kwargs_from_file("config.txt"))
