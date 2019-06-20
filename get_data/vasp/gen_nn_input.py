import numpy
from pymatgen.io.vasp import BSVasprun
from pymatgen.core.structure import IStructure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.electronic_structure.core import Spin
import json
from gen_vasp_input import struct_from_sgnum

from fractions import Fraction


def all_hspoints_from_file(file_path):
    arr = numpy.loadtxt(file_path, str)
    vfunc = numpy.vectorize(Fraction)
    return vfunc(arr).astype(float)


def write_nn_input(structure: IStructure,
                   data_dir: str,
                   sgnum: int,
                   index: int,
                   write_dir: str,
                   ):
    bsvasprun = BSVasprun(data_dir)

    kpoints = [tuple(vector) for vector in bsvasprun.actual_kpoints]  # sampled kpoints
    bandstructure = bsvasprun.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
    bands = bandstructure.bands[Spin.up].T

    kpoints_to_bands = dict(zip(kpoints, bands))

    all_hspoints = [tuple(vector) for vector in all_hspoints_from_file("all_hspoints.txt")]

    cur_hspoints = [tuple(kpoints) for kpoints in HighSymmKpath(structure).kpath["kpoints"].values()]

    cur_hspoints_to_bands = {
        kpoints:
            kpoints_to_bands[kpoints]
        for kpoints in cur_hspoints
    }

    all_hspoints_to_bands = {
        hspoints:
            cur_hspoints_to_bands[hspoints].tolist()
            if (hspoints in cur_hspoints_to_bands) else
            [0] * bands.shape[1]
        for hspoints in all_hspoints
    }

    all_bands = numpy.array(list(all_hspoints_to_bands.values())).T.tolist()

    input_dict = {"number": sgnum, "bands": all_bands}
    file_path = write_dir + "theo_input_data_{}.json".format(index)
    with open(file_path, "w") as file:
        json.dump(input_dict, file, indent=4)


if __name__ == "__main__":
    struct = struct_from_sgnum(
        sgnum=1,
        scaling_factor=4.7,
    )
    write_nn_input(struct, "vasprun_1_1.xml", 1, 0, "../data/theo_inputs/")
