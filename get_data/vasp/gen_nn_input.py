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


def all_hslabels_from_file(file_path):
    return numpy.loadtxt(file_path, str)


def write_nn_input_coord_based(structure: IStructure,
                               sgnum: int,
                               index: int,
                               all_hs_path: str,
                               vasprun_path: str,
                               write_dir: str, ):
    bsvasprun = BSVasprun(vasprun_path)

    kpoints = [tuple(vector) for vector in bsvasprun.actual_kpoints]  # (kpoints, 1)
    bandstructure = bsvasprun.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
    bands = bandstructure.bands[Spin.up].T  # (kpoints, bands)

    # connect two array
    kpoints_to_bands = dict(zip(kpoints, bands))

    # all possible high symmetry kpoints
    all_hspoints = [tuple(vector) for vector in all_hspoints_from_file(all_hs_path)]
    # high symmetry kpoints in this structure
    cur_hspoints = [tuple(kpoints) for kpoints in HighSymmKpath(structure).kpath["kpoints"].values()]

    # filter out bands of intermediate kpoints (cur_hspoints, bands)
    cur_hspoints_to_bands = {
        kpoints:
            kpoints_to_bands[kpoints]
        for kpoints in cur_hspoints
    }

    # pad zeros to missing high symmetry points (all_hspoints, bands)
    all_hspoints_to_bands = {
        hspoints:
            cur_hspoints_to_bands[hspoints].tolist()
            if (hspoints in cur_hspoints_to_bands) else
            [0] * bands.shape[1]
        for hspoints in all_hspoints
    }

    all_bands = numpy.array(list(all_hspoints_to_bands.values())).T.tolist()  # (bands, all_hspoints)

    input_dict = {"number": sgnum, "bands": all_bands}
    file_path = write_dir + "theo_input_data_{}_{}.json".format(sgnum, index)
    with open(file_path, "w") as file:
        json.dump(input_dict, file, indent=4)


def write_nn_input_label_based(sgnum: int,
                               index: int,
                               all_hs_path: str,
                               vasprun_path: str,
                               write_dir: str, ):
    bsvasprun = BSVasprun(vasprun_path)

    bandstructure = bsvasprun.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
    bands = bandstructure.bands[Spin.up].T  # (kpoints, bands)

    # all possible high symmetry labels
    all_hslabels = all_hslabels_from_file(all_hs_path)

    # filter out bands of intermediate kpoints (cur_hslabels, bands)
    cur_hslabels_to_bands = {}
    for kpath in bandstructure.branches:
        cur_hslabels_to_bands[kpath["name"].split("-")[0]] = bands[kpath["start_index"]]
        cur_hslabels_to_bands[kpath["name"].split("-")[1]] = bands[kpath["end_index"]]

    # pad zeros to missing high symmetry labels (all_hspoints, bands)
    all_hslabels_to_bands = {
        hslabels:
            cur_hslabels_to_bands[hslabels].tolist()
            if (hslabels in cur_hslabels_to_bands) else
            [0] * bands.shape[1]
        for hslabels in all_hslabels
    }

    all_bands = numpy.array(list(all_hslabels_to_bands.values())).T.tolist()  # (bands, all_hslabels)

    input_dict = {"number": sgnum, "bands": all_bands}
    file_path = write_dir + "theo_input_data_{}_{}.json".format(sgnum, index)
    with open(file_path, "w") as file:
        json.dump(input_dict, file, indent=4)


if __name__ == "__main__":
    struct = struct_from_sgnum(
        sgnum=1,
        scaling_factor=4.7,
    )
    write_nn_input_coord_based(
        structure=struct,
        sgnum=1,
        index=0,
        all_hs_path="all_hs_files/all_hspoints.txt",
        vasprun_path="vasprun.xml",
        write_dir="data_upload/coord_based/",
    )
    write_nn_input_label_based(
        sgnum=1,
        index=0,
        all_hs_path="all_hs_files/all_hslabels.txt",
        vasprun_path="vasprun.xml",
        write_dir="data_upload/label_based/",
    )
