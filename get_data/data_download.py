import pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.core import Spin
import multiprocessing
import numpy as np
import tqdm

m = MPRester("vI8phwVV3Ie6s4ke")


def job(mp_id):
    try:
        band = m.get_bandstructure_by_material_id(material_id="mp-" + str(mp_id), line_mode=True)
        if band is None:
            if verbose:
                print(str(mp_id) + " has no bands")
            return
    except IndexError:
        if verbose:
            print(str(mp_id) + " is not valid")
        return
    except pymatgen.ext.matproj.MPRestError:
        if verbose:
            print(str(mp_id) + " is not valid")
        return
    else:
        return m.get_doc("mp-{}".format(mp_id))["spacegroup"]


def errorfunc(e):
    raise e


def print_arr(arr, file_path):
    with open(file_path, "w") as file:
        for key, val in enumerate(arr):
            file.writelines("{} {}\n".format(key + 1, val))


n_thread = 300
total_start, total_end = 1, 500
verbose = False

def main():
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes=n_thread)

    counts = np.zeros(230, dtype=int)
    names = np.empty(230, dtype="U10")

    for spacegroup in tqdm.tqdm(pool.imap_unordered(job, range(total_end-total_start)), total=total_end-total_start):
        if spacegroup is None:
            continue
        number = spacegroup["number"]
        symbol = spacegroup["symbol"]
        counts[number] += 1
        names[number] = symbol

    pool.close()
    pool.join()

    print_arr(counts, "data/spacegroup_occurence_1-500.txt")
    print_arr(names, "data/spacegroup_conversion_1-500.txt")
    print('\nAll subprocesses done.\n')


if __name__ == "__main__":
    main()
