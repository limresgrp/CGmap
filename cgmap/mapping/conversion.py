import glob
import logging
import os
import re
import numpy as np
import pandas as pd
import MDAnalysis as mda

from pathlib import Path
from MDAnalysis.transformations import set_dimensions
from typing import List, Any, Optional

def format_resname(ffield: str):
    return f'{ffield.lower()}-resname'

def format_atomname(ffield: str):
    return f'{ffield.lower()}-atomname'


class Conversion():

    @property
    def resnames(self):
        if self._atom_resnames is None or self._atom_residcs is None:
            return None
        return self._atom_resnames[
            np.concatenate(
                ([0], np.where(self._atom_residcs[:-1] != self._atom_residcs[1:])[0] + 1)
                )
            ]
    
    @property
    def residcs(self):
        if self._atom_residcs is None:
            return None
        return np.arange(len(np.unique(self._atom_residcs)))
    
    @property
    def resnums(self):
        if self._atom_resnums is None:
            return None
        return self._atom_resnums[
            np.concatenate(
                ([0], np.where(self._atom_residcs[:-1] != self._atom_residcs[1:])[0] + 1)
                )
            ]

    def __init__(self, rootdirs: List[str] = None) -> None:    
        self._ffields: List[str] = []
        self._data: pd.DataFrame = None
        if rootdirs is None:
            rootdirs = []
        rootdirs.append(os.path.join(os.path.dirname(__file__), '..', 'conversion'))

        for rootdir in rootdirs:
            for filename in glob.glob(os.path.join(rootdir, '*.dat')):
                self.load(filename)
    
    def load(self, filename: str):
        self._load_filename(filename)
    
    def __call__(self, **kwargs: Any) -> Any:
        u = mda.Universe(
            kwargs.get("input"),
            *kwargs.get("inputtraj", []),
            **kwargs.get("extrakwargs", {})
        )

        sel = u.select_atoms(kwargs.get('selection', 'all'))

        atom_resnames = []
        atom_names    = []
        atom_types    = []
        atom_residcs  = []
        atom_resnums  = []
        atom_segids   = []

        for atom in sel.atoms:
            atom_resnames.append(atom.resname)
            atom_names.append(re.sub(r'^(\d+)\s*(.+)', r'\2\1', atom.name))
            atom_types.append(atom.type)
            atom_residcs.append(atom.resindex)
            atom_resnums.append(atom.resnum)
            atom_segids.append(atom.segid)
        
        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        self._atom_types = np.array(atom_types)
        self._atom_residcs = np.array(atom_residcs)
        self._atom_resnums = np.array(atom_resnums)
        self._atom_segids = np.array(atom_segids)

        atom_positions = []
        cell_dimensions = []
        
        try:
            for ts in u.trajectory:
                atom_positions.append(sel.positions)
                if ts.dimensions is not None:
                    cell_dimensions.append(ts.dimensions)
        except ValueError as e:
            self.logger.error(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")

        self._atom_positions =  np.stack(atom_positions, axis=0)
        self._cell = np.stack(cell_dimensions, axis=0) if len(cell_dimensions) > 0 else None

        converted_atom_resnames = []
        converted_atom_names    = []
        
        for atom_resname, atom_name in zip(self._atom_resnames, self._atom_names):
            try:
                converted_atom_resname, converted_atom_name = self._convert(
                    atom_resname,
                    atom_name,
                    kwargs.get("ffield"),
                )
            except Exception as e:
                logging.warning(e)
                converted_atom_resname = atom_resname
                converted_atom_name = atom_name
            converted_atom_resnames.append(converted_atom_resname)
            converted_atom_names.append(converted_atom_name)
        
        converted_atom_resnames = np.array(converted_atom_resnames)
        converted_atom_names    = np.array(converted_atom_names)

        sorted_converted_atom_resnames, sorted_converted_atom_names = self._sort_atoms(
            converted_atom_resnames,
            converted_atom_names,
        )

        last_rn_old, last_rn_new = self._atom_resnames[0], sorted_converted_atom_resnames[0]
        last_ri = self._atom_residcs[0]
        updated_atom_residcs = np.copy(self._atom_residcs)
        updted_atom_resnums = np.copy(self._atom_resnums)
        new_residue = True
        old_starting_rn = None
        for idx, (rn_old, rn_new, ri) in enumerate(zip(
            self._atom_resnames[1:], sorted_converted_atom_resnames[1:], self._atom_residcs[1:].copy()
        )):
            if rn_new != last_rn_new and ri == last_ri:
                updated_atom_residcs[idx+1:] += 1
                updted_atom_resnums[idx+1:] += 1
            elif rn_old != last_rn_old and rn_new == last_rn_new:
                if new_residue:
                    old_starting_rn = last_rn_old
                    new_residue = False
                if rn_old != old_starting_rn:
                    updated_atom_residcs[idx+1:] -= 1
                    updted_atom_resnums[idx+1:]    -= 1
                else:
                    new_residue = True
            last_rn_old = rn_old
            last_rn_new = rn_new
            last_ri = ri

        self._atom_resnames = sorted_converted_atom_resnames
        self._atom_names    = sorted_converted_atom_names
        self._atom_residcs = updated_atom_residcs - updated_atom_residcs.min()
        self._atom_resnums = updted_atom_resnums

        self._n_atoms = len(self._atom_names)
        self._n_residues = len(np.unique(self._atom_residcs))
        self._n_segments = len(np.unique(self._atom_segids))

        p = Path(kwargs.get("input"))
        out_filename = str(Path(p.parent, p.stem + f'.{kwargs.get("ffield").lower()}' + p.suffix))
        self._save(filename=out_filename, traj_format=kwargs.get("trajformat", 'xtc'))
        return self
    
    def _sort_atoms(self, converted_atom_resnames, converted_atom_names):
        import pandas as pd

        df = pd.DataFrame({
            'atom_resnames': converted_atom_resnames,
            'atom_names': converted_atom_names,
            'atom_resindices': self._atom_residcs,
        })

        sorting_idcs = df.sort_values(by=['atom_resindices', 'atom_resnames']).index

        self._atom_resnames = self._atom_resnames[sorting_idcs]
        self._atom_names = self._atom_names[sorting_idcs]
        self._atom_types = self._atom_types[sorting_idcs]
        self._atom_residcs = self._atom_residcs[sorting_idcs]
        self._atom_resnums = self._atom_resnums[sorting_idcs]
        self._atom_segids = self._atom_segids[sorting_idcs]
        self._atom_positions = self._atom_positions[:, sorting_idcs]
        
        return converted_atom_resnames[sorting_idcs], converted_atom_names[sorting_idcs]
    
    def _save(self, filename: Optional[str] = None, traj_format: Optional[str] = None):
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        u = mda.Universe.empty(
            n_atoms =       self._n_atoms,
            n_residues =    self._n_residues,
            n_segments =    self._n_segments,
            atom_resindex = self._atom_residcs,
            trajectory =    True, # necessary for adding coordinates
        )
        u.add_TopologyAttr('names',    self._atom_names)
        u.add_TopologyAttr('types',    self._atom_types)
        u.add_TopologyAttr('resnames', self.resnames)
        u.add_TopologyAttr('resid',    self.residcs)
        u.add_TopologyAttr('resnum',   self.resnums)
        u.add_TopologyAttr('chainID',  self._atom_segids)
        u.load_new(self._atom_positions, order='fac')
        if self._cell is not None:
            transform = set_dimensions(self._cell)
            u.trajectory.add_transformations(transform)
        selection = u.select_atoms('all')
        with mda.Writer(filename, n_atoms=u.atoms.n_atoms) as w:
            w.write(selection.atoms)

        if len(u.trajectory) > 1:
            if traj_format is None:
                traj_format = self.config.get('outputtraj', 'xtc')
            filename_traj = str(Path(filename).with_suffix('.' + traj_format))
            with mda.Writer(filename_traj, n_atoms=u.atoms.n_atoms) as w_traj:
                for ts in u.trajectory:  # Skip the first frame as it's already saved
                    w_traj.write(selection.atoms)

    def _convert(self, resname: str, atomname: str, ffield: str, *args: Any, **kwargs: Any) -> Any:
        converted_resname, converted_atomname = None, None
        df = self._data
        for src_ffield in self.ffields:
            results = df[(df[format_resname(src_ffield)] == resname) & (df[format_atomname(src_ffield)] == atomname)]
            if len(results) == 0:
                continue
            if len(results) > 1:
                raise Exception(f'Found duplicate naming in conversion mapping file(s) for {resname} {atomname}.')
            try:
                converted_resname, converted_atomname = results[[format_resname(ffield), format_atomname(ffield)]].values[0]
                break
            except:
                raise Exception(f'No information found in conversion mapping file(s) regarding force field {ffield} for {resname} {atomname}.')
        if converted_resname is None:
            raise Exception(f'No information found in conversion mapping file(s) about {resname} {atomname}.')
        return converted_resname, converted_atomname
    
    def _load_filename(self, filename):
        columns = None
        rows = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                if columns is None:
                    ffields = line.split()
                    self.ffields = ffields
                    columns = []
                    for ffield in ffields:
                        columns.extend([
                            format_resname(ffield),
                            format_atomname(ffield),
                        ])
                    continue
                if line.strip().startswith('!'):
                    continue
                elems = line.split()
                assert len(elems) == len(columns)
                
                row = {c: e for c, e in zip(columns, elems)}
                rows.append(row)
        
        data = pd.DataFrame(data=rows, columns=columns)

        if self._data is None:
            self._data = data
        else:
            self._data = pd.concat([self._data, data])

        return self