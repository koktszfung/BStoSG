import os
import numpy
from gen_vasp_input import struct_from_sgnum, write_vasp_input
from gen_nn_input import write_nn_input_label_based, write_nn_input_coord_based
from plot_vasp_output import plot_dos, plot_bs
import argparse


def vasp(sgnum: int,
         index: int,
         node: int = 4):
    try:
        print("\tvasp {} start".format(index))
        os.system("mpiexec -n {} vasp-544-s > vasp.out".format(node))
    except Exception as e:
        print("\tskipped due to {}".format(e))
        return
    else:
        os.system("cp vasprun.xml vaspruns/vasprun_{}_{}".format(sgnum, index))
        print("\tvasp end")


def gen_bs_from_sgnum(kpath_division: int,
                      sgnum: int,
                      index: int = None,
                      is_plot: bool = False):
    struct = struct_from_sgnum(
        sgnum=sgnum,
        scaling_factor=4,
        lattice_vectors=None,
        init_coords=numpy.random.rand(1, 3),
        species=None,
    )

    write_vasp_input(
        structure=struct,
        kpath_division=kpath_division,
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

    if is_plot:
        # plot_dos("vasprun.xml")
        plot_bs("vasprun.xml")


def main(kpath_division: int,
         sgnum: int,
         start: int = 0,
         total: int = 1,
         is_plot: bool = False):
    print("main {} start".format(sgnum))
    for i in range(start, start + total - 1):
        gen_bs_from_sgnum(
            kpath_division=kpath_division,
            sgnum=sgnum,
            index=i,
            is_plot=False,
        )
    gen_bs_from_sgnum(
        kpath_division=kpath_division,
        sgnum=sgnum,
        index=start + total - 1,
        is_plot=is_plot,
    )
    print("main end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NN Input Generation")
    parser.add_argument("-d", "--division", type=int, help="number of kpoints on a kpath")
    parser.add_argument("-g", "--sgnum", type=int, help="spacegroup number 1~230")
    parser.add_argument("-s", "--start", type=int, default=0, help="starting index")
    parser.add_argument("-t", "--total", type=int, default=1, help="total number of runs")
    parser.add_argument("-p", "--plot", action="store_true", help="plot the last bandstructure")

    args = parser.parse_args()

    main(args.division, args.sgnum, args.start, args.total, args.plot)

