from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter


def plot_dos(data_dir: str):
    v = Vasprun(data_dir)
    tdos = v.tdos
    plt = DosPlotter()
    plt.add_dos("Total DOS", tdos)
    plt.show()


def plot_bs(data_dir: str):
    v = BSVasprun(data_dir)
    bs = v.get_band_structure(kpoints_filename="KPOINTS", line_mode=True)
    plt = BSPlotter(bs)
    plt.show()


if __name__ == "__main__":
    plot_dos("vasprun.xml")
    plot_bs("vasprun.xml")
