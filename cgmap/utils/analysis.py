from typing import List, Optional, Tuple, Union
import numpy as np
import MDAnalysis as mda
import seaborn as sns

from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSD
from MDAnalysis.analysis.dihedrals import Ramachandran
from matplotlib import pyplot as plt

def align_and_rmsd(
    universe: mda.Universe,
    ref_universe: Optional[mda.Universe] = None,
    align_on: str = 'all',
    rmsd_on: Optional[Union[str, List[str]]] = None,
    ref_frame: Union[int, str] = 0,
    figsize: Tuple[int] = (8, 8),
    trajslice: Optional[slice] = None,
):
    if isinstance(rmsd_on, List):
        groupselections = rmsd_on
    elif isinstance(rmsd_on, str):
        groupselections = [rmsd_on]
    else:
        groupselections = [align_on]
    
    if trajslice is None:
        trajslice = slice(None, None, None)
    
    if isinstance(ref_frame, str):
        if ref_frame == 'average':
            if ref_universe is None:
                ref_universe = universe

            aligner = align.AlignTraj(ref_universe, ref_universe, select=align_on, in_memory=True).run()
            selection = ref_universe.select_atoms('all')
            pos = []
            for ts in ref_universe.trajectory[trajslice]:
                pos.append(selection.positions)
            average_pos = np.mean(np.stack(pos, axis=0), axis=0, keepdims=True)
            
            ref_universe = mda.Universe(universe.filename)
            ref_universe.load_new(average_pos, order='fac')
            selection = ref_universe.select_atoms('protein')
            with mda.Writer('test.pdb', n_atoms=ref_universe.atoms.n_atoms) as w:
                w.write(selection.atoms)
            ref_frame = 0
        else:
            raise Exception(f"param 'ref_frame' has an invalid value: {ref_frame}. Allowed values are any integer or 'average'")
    else:
        if ref_universe is None:
            ref_frame = np.arange(len(universe.trajectory))[trajslice][ref_frame]
        else:
            ref_frame = np.arange(len(ref_universe.trajectory))[trajslice][ref_frame]
    
    R = RMSD(
        universe,
        ref_universe,
        select=align_on,
        groupselections=groupselections,
        ref_frame=ref_frame,
    )
    result = R.run()

    ts = result.results.rmsd[trajslice, 1] / 1000 # from ps to ns
    ts -= ts.min()
    rmsd = result.results.rmsd[trajslice, 3:]

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax  = fig.add_subplot(111)
    ax.plot(ts, rmsd)
    return fig

def ramachandran(
    selection,
    trajslice: Optional[slice] = None,
    ref_selection = None,
    ref_trajslice: Optional[slice] = None,
    thresh: float = 0.05,
    levels: int = 10,
    figsize: Tuple[int] = (8, 8),
):
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax  = fig.add_subplot(111)
    plt.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.02,
        hspace=0.1,
    )

    def normalize(x):
        x = x / 180 * np.pi
        if np.any(x > np.pi):
            x -= np.pi
        return x
    
    if trajslice is None:
        trajslice = slice(None, None, None)

    R = Ramachandran(selection).run()
    phi_psi = normalize(R.results.angles[trajslice].reshape(-1, 2))
    sns.kdeplot(
        x=phi_psi[:, 0],
        y=phi_psi[:, 1],
        cmap=sns.color_palette(f"blend:#EEE,{sns.color_palette().as_hex()[0]}", as_cmap=True),
        fill=True, thresh=thresh, ax=ax, levels=levels, bw_method=0.18,
    )
    sns.kdeplot(
        x=phi_psi[:, 0],
        y=phi_psi[:, 1],
        color=sns.color_palette()[0],
        fill=False, thresh=thresh, ax=ax, levels=levels, linewidths=0.1, bw_method=0.18
    )

    if ref_selection is not None:
        if ref_trajslice is None:
            ref_trajslice = slice(None, None, None)
        
        ref_R = Ramachandran(ref_selection).run()
        ref_phi_psi = normalize(ref_R.results.angles[ref_trajslice].reshape(-1, 2))

        sns.kdeplot(
            x=ref_phi_psi[:, 0],
            y=ref_phi_psi[:, 1],
            color=sns.color_palette()[1],
            fill=False, thresh=thresh, ax=ax, levels=levels, linewidths=0.5, bw=0.18
        )

    ax.set_xlim(xmin=-np.pi, xmax=np.pi)
    ax.set_ylim(ymin=-np.pi, ymax=np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi$/2', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi/2$', r'$\pi$'])

    return fig