from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter
import argparse
import xml


def plot_dos(vasprun_path: str):
    try:
        v = Vasprun(vasprun_path)
    except xml.etree.ElementTree.ParseError:
        print("\tskipped due to parse error")
        return
    tdos = v.tdos
    plt = DosPlotter()
    plt.add_dos("Total DOS", tdos)
    plt.show()


def plot_bs(vasprun_path: str):
    try:
        v = BSVasprun(vasprun_path)
    except xml.etree.ElementTree.ParseError:
        print("\tskipped due to parse error")
        return
    bs = v.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
    plt = BSPlotter(bs)
    plt.show()


def main(vasprun_path: str):
    plot_dos(vasprun_path)
    plot_bs(vasprun_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NN Input Generation")
    parser.add_argument("-p", "--path", type=str, default="vasprun.xml", help="file path to vasprun.xml")

    args = parser.parse_args()

    main(args.path)
