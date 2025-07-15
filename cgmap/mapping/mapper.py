from __future__ import annotations

import copy
import glob
import logging
import re
import os
import sys
import yaml
import traceback
import itertools
import numpy as np
import MDAnalysis as mda
import random

from os.path import dirname, basename
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple
from MDAnalysis.core.groups import Atom
from cgmap.mapping.utils import parse_slice
from .bead import BeadMappingSettings, BeadMappingAtomSettings, Bead
from cgmap.utils import DataDict
from cgmap.utils.atomType import get_type_from_name
from ase.geometry import cellpar_to_cell

import warnings
warnings.filterwarnings("ignore")


class Mapper():

    _map_func: Callable
    _map_impl_func: Callable

    u: mda.Universe = None
    trajectory = None
    bead_types_filename: str = "bead_types.yaml"

    _weigth_based_on: str = 'mass'
    _atom2bead: Dict[str, str] = {}
    _bead2atom: Dict[str, List[str]] = {}
    _bead_types: Dict[str, int] = {}
    _bead_mapping_settings: Dict[str, BeadMappingSettings] = {}
    _bead_cm: Dict[str, str] = {}

    _appearence_idnames: Dict[str, str] = {}
    _incomplete_beads: List[Bead] = []
    _complete_beads:   List[Bead] = []
    _ordered_beads:    List[Bead] = []

    _max_bead_all_atoms: int = 0
    _max_bead_reconstructed_atoms: int = 0
    _n_beads: int = 0
    _n_atoms: int = 0

    _atom_positions: np.ndarray = None
    _atom_idnames: np.ndarray = None
    _atom_resnames: np.ndarray = None
    _atom_names: np.ndarray = None
    _atom_resindices: np.ndarray = None
    _atom_resnums: np.ndarray = None
    _atom_segindices: np.ndarray = None
    _atom_forces: np.ndarray = None

    _atom2idcs_dict: Dict[str, int] = {}

    _bead_positions: np.ndarray = None
    _bead_idnames: np.ndarray = None
    _bead_resnames: np.ndarray = None
    _bead_names: np.ndarray = None
    _bead_resindices: np.ndarray = None
    _bead_resnums: np.ndarray = None
    _bead_segids: np.ndarray = None
    _bead_optim_forces = None

    _bead2atom_idcs: np.ndarray = None
    _bead2atom_weights: np.ndarray = None
    
    _cell: np.ndarray = None
    _pbc: np.ndarray = np.array([True, True, True])

    
    # Properties of mapping

    @property
    def input_filename(self):
        return self.config.get('input', None)
    
    @property
    def bead_reconstructed_size(self):
        return self._max_bead_reconstructed_atoms
    
    @property
    def bead_all_size(self):
        return self._max_bead_all_atoms
    
    @property
    def bead_types_dict(self):
        return self._bead_types
    
    @property
    def n_bead_types(self):
        return len(self._bead_types)

    # Properties of mapped instance

    @property
    def num_frames(self):
        if self._atom_positions is None:
            return None
        return len(self._atom_positions)
    
    @property
    def bead2atom_idcs(self):
        if self._bead2atom_idcs is None:
            return None
        return [b2a_id[b2a_id != -1].tolist() for b2a_id in self._bead2atom_idcs]
    
    @property
    def resnames(self):
        if self._atom_resnames is None or self._atom_resindices is None:
            return None
        return self._atom_resnames[
            np.concatenate(
                ([0], np.where(self._atom_resindices[:-1] != self._atom_resindices[1:])[0] + 1)
                )
            ]
    
    @property
    def resindices(self):
        if self._atom_resindices is None:
            return None
        return np.arange(len(np.unique(self._atom_resindices)))
    
    @property
    def resnumbers(self):
        if self._atom_resnums is None:
            return None
        return self._atom_resnums[
            np.concatenate(
                ([0], np.where(self._atom_resindices[:-1] != self._atom_resindices[1:])[0] + 1)
                )
            ]
    
    @property
    def num_residues(self):
        if self._atom_segindices is None or self._atom_resindices is None:
            return None
        chain_residue_names = []
        for chain, resindex in zip(self._atom_segindices, self._atom_resindices):
            chain_residue_names.append(f'{str(chain)}_{str(resindex)}')
        return len(np.unique(chain_residue_names))
    
    @property
    def num_atoms(self):
        if self._atom_names is None:
            return None
        return len(self._atom_names)

    @property
    def atom_types(self):
        if self._atom_names is None:
            return None
        return np.array([get_type_from_name(name) for name in self._atom_names])
    
    @property
    def atom_forces(self):
        if self._atom_forces is None:
            return None
        if np.all(np.isnan(self._atom_forces)):
            return None
        return np.nan_to_num(self._atom_forces)

    @property
    def num_beads(self):
        if self._bead_idnames is None:
            return None
        return len(self._bead_idnames)
    
    @property
    def bead_types(self):
        if self._bead_idnames is None:
            return None
        return np.array([self._bead_types[idname] for idname in self._bead_idnames])
    
    @property
    def bead_resindices(self):
        if self._bead_resindices is None:
            return None
        _, c = np.unique(self._bead_resindices, return_counts=True)
        return np.repeat(np.arange(len(c)), c)
    
    @property
    def bead_optim_forces(self):
        MAX_FRAMES = 1000
        try:
            from aggforce import linearmap as lm
            from aggforce import agg as ag
            if self._bead_optim_forces is None and self._atom_forces is not None:
                idcs = self.bead2atom_idcs
                idcs = [[i for i in id if ~np.any(np.isnan(self._atom_forces[0, i]))] for id in idcs]
                cmap = lm.LinearMap(idcs, n_fg_sites=self.num_atoms)
                # constraints = cf.guess_pairwise_constraints(self._atom_positions, threshold=1e-3)
                constraints = None

                if self.num_frames > MAX_FRAMES:
                    # Sample uniformly from rows of A
                    sampled_rows = np.sort(np.random.choice(self.num_frames, size=MAX_FRAMES, replace=False))
                    atom_forces = self.atom_forces[sampled_rows]
                else:
                    # Sample from A directly
                    atom_forces = self.atom_forces

                self.logger.info(f"Optimizing the mapping of atomistic forces to CG...")
                self._bead_optim_forces = ag.project_forces(
                    method=ag.linearmap.qp_linear_map_per_cg_site, # ag.linearmap.constraint_aware_uni_map
                    xyz=None,
                    forces=atom_forces,
                    config_mapping=cmap,
                    constrained_inds=constraints,
                )
                self._bead_optim_forces["projected_forces"] = np.einsum('ijk,bj->ibk', self.atom_forces, self.bead2atom_forces_weights)
            return self._bead_optim_forces
        except:
            return None
    
    @property
    def bead_forces(self):
        if self.bead_optim_forces is None:
            return None
        return self.bead_optim_forces["projected_forces"]
    
    @property
    def bead2atom_forces_weights(self):
        if self.bead_optim_forces is None:
            return None
        return self._bead_optim_forces["map"].standard_matrix
    
    @property
    def dataset(self):
        return {k: v for k, v in {
            DataDict.NUM_RESIDUES:    self.num_residues,
            DataDict.RESNAMES:        self.resnames,
            DataDict.RESIDCS:         self.resindices,
            DataDict.RESNUMBERS:      self.resnumbers,

            DataDict.NUM_ATOMS:       self.num_atoms,
            DataDict.ATOM_POSITION:   self._atom_positions,
            DataDict.ATOM_RESNAMES:   self._atom_resnames,
            DataDict.ATOM_NAMES:      self._atom_names,
            DataDict.ATOM_TYPES:      self.atom_types,
            DataDict.ATOM_RESIDCS:    self._atom_resindices,
            DataDict.ATOM_RESNUMBERS: self._atom_resnums,
            DataDict.ATOM_SEGIDS:     self._atom_segindices,
            DataDict.ATOM_FORCES:     self.atom_forces,

            DataDict.NUM_BEADS:       self.num_beads,
            DataDict.BEAD_POSITION:   self._bead_positions,
            DataDict.BEAD_IDNAMES:    self._bead_idnames,
            DataDict.BEAD_RESNAMES:   self._bead_resnames,
            DataDict.BEAD_NAMES:      self._bead_names,
            DataDict.BEAD_TYPES:      self.bead_types,
            DataDict.BEAD_RESIDCS:    self.bead_resindices,
            DataDict.BEAD_RESNUMBERS: self._bead_resnums,
            DataDict.BEAD_SEGIDS:     self._bead_segids,
            DataDict.BEAD_FORCES:     self.bead_forces,

            DataDict.CELL:            self._cell,
            DataDict.PBC:             self._pbc,

            DataDict.BEAD2ATOM_IDCS:      self._bead2atom_idcs,
            DataDict.BEAD2ATOM_WEIGHTS:   self._bead2atom_weights,
        }.items() if v is not None}

    def __init__(self, args_dict, logger = None) -> None:
        if "bead_types_filename" in args_dict:
            self.bead_types_filename = args_dict.get("bead_types_filename")
        self.logger = logger if logger is not None else logging.getLogger()

        config: Dict[str, str] = dict()

        yaml_config = args_dict.pop("config", None)
        if yaml_config is not None:
            config.update(yaml.safe_load(Path(yaml_config).read_text()))
        
        args_dict = {key: value for key, value in args_dict.items() if value is not None}
        config.update(args_dict)

        if config.get("trajslice", None) is not None:
            config["trajslice"] = parse_slice(config["trajslice"])

        input = config.get("input")
        inputtraj = config.get("inputtraj", None)

        if os.path.isdir(input):
            input_folder = input
            input_format = config.get("inputformat", "*")
            filter = config.get("filter", None)
            input_basenames = None
            if filter is not None:
                with open(filter, 'r') as f:
                    input_basenames = [line.strip() for line in f.readlines()]
            self.input_filenames = [
                fname for fname in
                list(glob.glob(os.path.join(input_folder, f"*.{input_format}")))
                if input_basenames is None
                or basename(fname).replace(f".{input_format}", "") in input_basenames
            ]
        else:
            input_folder = None
            input_filename = input
            self.input_filenames = [input_filename]
        
        if inputtraj is None:
            self.input_trajnames = [None for _ in range(len(self.input_filenames))]
        else:
            if os.path.isdir(inputtraj):
                traj_folder = inputtraj
                traj_format = config.get("trajformat", "*")
                filter = config.get("filter", None)
                input_basenames = None
                if filter is not None:
                    with open(filter, 'r') as f:
                        input_basenames = [line.strip() for line in f.readlines()]
                self.input_trajnames = [
                    fname for fname in
                    list(glob.glob(os.path.join(traj_folder, f"*.{traj_format}")))
                    if input_basenames is None
                    or basename(fname).replace(f".{traj_format}", "") in input_basenames
                ]
            else:
                self.input_trajnames = [inputtraj]

        self._map_impl_func = self._map_impl if config.get("isatomistic", False) else self._map_impl_cg

        self.config = config
        self._weigth_based_on = config.get("weigth_based_on", "mass")
        self.trajectory = None

        mapping_folder = config.get("mapping", None)
        if mapping_folder is None:
            raise Exception(
                """
                You must provide the mapping folder.
                Add 'mapping_folder: name-of-mapping-folder' in the config file.
                mappings are specified in a mapping folder inside 'heqbm/data/'
                """
            )
        if os.path.isabs(mapping_folder):
            self._root_mapping = mapping_folder
        else:
            self._root_mapping = os.path.join(os.path.dirname(__file__), '..', 'data', mapping_folder)
        
        # Iterate configuration files and load all mappings
        self._clear_mappings()
        self._load_mappings()
    
    def __call__(self, **kwargs) -> Generator[Tuple[Mapper, str]]:
        for input_filename, input_trajname in self.iterate():
            p = Path(input_filename)
            output_filename = str(Path(kwargs.get('output', self.config.get('output', p.parent)), p.stem + '.data' + '.npz'))
            if os.path.isfile(output_filename):
                continue
            try:
                yield self.map_impl(input_filename, input_trajname), output_filename
            except Exception as e:
                self.logger.error(f"Error processing file {input_filename}: {e}")
                self.logger.error(traceback.format_exc())
    
    def iterate(self, *args, **kwargs):
        for input_filename, input_trajname in zip(self.input_filenames, self.input_trajnames):
            yield input_filename, input_trajname
    
    def map(self, index: int = 0):
        input_filename = self.input_filenames[index]
        input_trajname = self.input_trajnames[index]
        return self.map_impl(input_filename, input_trajname)
    
    def map_impl(self, input_filename, input_trajname):
        self.config["input"] = input_filename
        self.config["inputtraj"] = [input_trajname] if input_trajname is not None else []
        self.logger.info(f"Mapping {input_filename} structure")

        self._clear_map()

        u = mda.Universe(
            self.config.get("input"),
            *self.config.get("inputtraj", []),
            **self.config.get("extrakwargs", {}))
        self.universe = u
        
        trajslice = self.config.get("trajslice", None)
        if trajslice is None:
            self.trajectory = u.trajectory
        else:
            self.trajectory = u.trajectory[trajslice]

        selection = self.config.get("selection", "all")
        self.selection = u.select_atoms(selection)

        self._map_impl_func()
        return self

    def __len__(self):
        if self.trajectory is None:
            return 0
        return len(self.trajectory)

    def _clear_mappings(self):
        self._atom2bead.clear()
        self._bead2atom.clear()
        self._bead_types.clear()
        self._bead_mapping_settings.clear()
        
        self._max_bead_reconstructed_atoms: int = 0
        self._max_bead_all_atoms: int = 0

    def _load_mappings(self, bms_class = BeadMappingSettings, bmas_class = BeadMappingAtomSettings):
        # Load bead types file, if existent.
        # It contains the bead type to identify each bead inside the NN
        # Different beads could have the same bead type (for example, all backbone beads could use the same bead type)
        bead_types_filename = os.path.join(self._root_mapping, self.bead_types_filename)
        if os.path.isfile(bead_types_filename):
            bead_types_conf: dict = yaml.safe_load(Path(bead_types_filename).read_text())
        else:
            bead_types_conf: dict = dict()
        
        # Iterate mapping files -> 1 mapping file = 1 residue mapping
        for filename in os.listdir(self._root_mapping):
            if filename.startswith("bead_types"):
                continue
            
            _conf_bead2atom = OrderedDict({})

            conf: dict = OrderedDict(yaml.safe_load(Path(os.path.join(self._root_mapping, filename)).read_text()))
            mol: str = conf.get("molecule")
            atoms_settings: dict[str, str] = conf.get("atoms")
            for atom_names, bead_settings_str in atoms_settings.items():
                all_bead_settings = bead_settings_str.split()
                bead_names = all_bead_settings[0].split(',')
                
                for i, bn in enumerate(bead_names):
                    bead_idname = DataDict.STR_SEPARATOR.join([mol, bn])
                    bead_settings = [x.split(',')[i] for x in all_bead_settings[1:]]
                    bms = self._bead_mapping_settings.get(bead_idname, bms_class(bead_idname))

                    atom_idname_alternatives = []
                    for atom in atom_names.strip().split(','):
                        atom_idname = DataDict.STR_SEPARATOR.join([mol, atom])
                        atom_idname_alternatives.append(atom_idname)
                        atom2bead_list = self._atom2bead.get(atom_idname, [])
                        atom2bead_list.append(bead_idname)
                        self._atom2bead[atom_idname] = atom2bead_list
                        
                    bmas = bms.get_bmas_by_atom_idname(atom_idname_alternatives)
                    if bmas is None:
                        bmas = bmas_class(
                            bead_settings,
                            bead_idname,
                            atom_idname_alternatives,
                            num_shared_beads=len(bead_names)
                        )
                        bms.add_atom_settings(bmas)
                        self._bead_mapping_settings[bead_idname] = bms
                    
                    bead2atom: List[str] = _conf_bead2atom.get(bead_idname, [])
                    if len(bead2atom) == 0 and bead_idname not in self._bead_types:
                        bead_type = bead_types_conf.get(bead_idname, max(bead_types_conf.values(), default=-1)+1)
                        bead_types_conf[bead_idname] = bead_type
                        self._bead_types[bead_idname] = bead_type
                    atom_idname_alternatives = tuple(atom_idname_alternatives)
                    assert atom_idname_alternatives not in bead2atom, f"{atom_names} is already present in bead {bead_idname}. Duplicate mapping."
                    bead2atom.append(atom_idname_alternatives)
                    _conf_bead2atom[bead_idname] = bead2atom
            
            bead_names_ordered = []
            for bead_name in itertools.chain(*self._atom2bead.values()):
                if bead_name not in bead_names_ordered:
                    bead_names_ordered.append(bead_name)
            for k, v in self._atom2bead.items():
                new_v = []
                for bns in bead_names_ordered:
                    if bns in v:
                        new_v.append(bns)
                self._atom2bead[k] = new_v
            
            for bms in self._bead_mapping_settings.values():
                bms.complete()
                self._max_bead_reconstructed_atoms = max(self._max_bead_reconstructed_atoms, bms.bead_reconstructed_size)
                self._max_bead_all_atoms = max(self._max_bead_all_atoms, bms.bead_all_size)
                
            for bead_idname, bead2atom in _conf_bead2atom.items():
                _bead2atom = self._bead2atom.get(bead_idname, [])
                _bead2atom.append(bead2atom)
                self._bead2atom[bead_idname] = _bead2atom
        
        with open(bead_types_filename, 'w') as outfile:
            yaml.dump(bead_types_conf, outfile, default_flow_style=False)
        
        for k, b2a in self._bead2atom.items():
            len_base = len(b2a[0])
            assert all([len(v) == len_base for v in b2a]), f"Configurations for bead type {k} have different number of atoms"

    def _load_extra_mappings(self, bead_splits, atom_name, bead_idname):
        return

    def _clear_map(self):
        self._appearence_idnames.clear()
        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()
        self._n_beads = 0
        self._n_atoms = 0
        self._bead_optim_forces = None
    
    def _map_impl_cg(
        self,
        conf_id: int = 0 # Which configuration to select for naming atoms
    ):
        self._bead_resnames =   self.selection.atoms.resnames
        self._bead_names =      self.selection.atoms.names
        self._bead_idnames = np.array([
            f"{resname}{DataDict.STR_SEPARATOR}{bead}"
            for resname, bead in zip(self._bead_resnames, self._bead_names)
        ])
        self._bead_resindices = self.selection.atoms.resindices
        self._bead_resnums =    self.selection.atoms.resnums
        self._bead_segids =     self.selection.atoms.segids

        bead_positions = []
        cell_dimensions = []
        
        try:
            for ts in self.trajectory:
                bead_pos = self.selection.positions
                if ts.dimensions is not None:
                    # Convert MDAnalysis ts.dimension ([lx, ly, lz, alpha, beta, gamma]) to 3x3 matrix
                    cell_dimensions.append(cellpar_to_cell(ts.dimensions))
                bead_positions.append(bead_pos)
        except ValueError as e:
            self.logger.error(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")

        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._cell = np.stack(cell_dimensions, axis=0) if len(cell_dimensions) > 0 else None

        atom_resnames =   []
        atom_names =      []
        atom_idnames =    []
        atom_resindices = []
        atom_resnums =    []
        atom_segindices = []

        atom_idnames_missing_multiplicity = {}
        atom_idnames2index = {}
        for h, bead_idname in enumerate(self._bead_idnames):
            bead = self._create_bead(bead_idname)
            for atom_idname in bead._config_ordered_atom_idnames[conf_id]:
                if isinstance(atom_idname, np.ndarray):
                    atom_idname = atom_idname[0].item()
                if atom_idnames_missing_multiplicity.get(atom_idname, 0) == 0:
                    atom_idnames.append(atom_idname)
                    atom_resname, atom_name = atom_idname.split(DataDict.STR_SEPARATOR)
                    atom_resnames.append(atom_resname)
                    atom_names.append(atom_name)
                    atom_resindices.append(self._bead_resindices[h])
                    atom_resnums.append(self._bead_resnums[h])
                    atom_segindices.append(self._bead_segids[h])

                    atom_idnames_missing_multiplicity[atom_idname] = len(np.unique(self._atom2bead[atom_idname])) - 1
                    atom_index = None
                else:
                    atom_idnames_missing_multiplicity[atom_idname] -= 1
                    atom_index = atom_idnames2index.get(atom_idname)

                atom_index, _ = self._update_bead(bead, atom_idname, atom_index=atom_index)
                atom_idnames2index[atom_idname] = atom_index
            
            self._check_bead_completeness(bead)

        self._atom_idnames    = np.array(atom_idnames)
        self._atom_resnames   = np.array(atom_resnames)
        self._atom_names      = np.array(atom_names)
        self._atom_resindices = np.array(atom_resindices)
        self._atom_resnums    = np.array(atom_resnums)
        self._atom_segindices = np.array(atom_segindices)

        batch, _, xyz = self._bead_positions.shape
        self._atom_positions =  np.empty((batch, len(self._atom_names), xyz), dtype=self._bead_positions.dtype)
        self._atom_positions[:] = np.nan

        self._compute_bead2atom_feats()
        self._compute_extra_map_impl()
        return self
            
    def _map_impl(self):
        
        bead_idnames =  []
        bead_resnames = []
        bead_names =    []
        bead_residcs =  []
        bead_resnums =  []
        bead_segids =   []

        last_resindex = -1
        for atom in self.selection.atoms:
            try:
                atom_idname = DataDict.STR_SEPARATOR.join([atom.resname, re.sub(r'^(\d+)\s*(.+)', r'\2\1', atom.name)])

                # This check is necessary to complete beads on residue change.
                # This allows having beads with incomplete atoms that do not remain pending
                if atom.resindex > last_resindex:
                    while len(self._incomplete_beads) > 0:
                        self._complete_bead(self._incomplete_beads.pop())
                    last_resindex = atom.resindex
                
                # Get existing beads which contain the current atom.
                # If no beads exist, create a new one.
                # Multiple beads could be retrieved, in case an atom is shared among multiple beads
                beads = self._get_incomplete_bead_from_atom_idname(atom_idname)

                # Iterate the retrieved beads.
                atom_index = None
                for bead in beads:
                    # If the bead is newly created, record its info.
                    if bead.is_newly_created:
                        bead_idnames.append(bead.idname)
                        bead_resnames.append(bead.resname)
                        bead_names.append(bead.name)
                        bead_residcs.append(atom.resindex)
                        bead_resnums.append(atom.resnum)
                        bead_segids.append(atom.segid)
                    
                    # Update the bead object with the current atom properties
                    atom_index, idcs_to_update = self._update_bead(bead, atom_idname, atom=atom, atom_index=atom_index)

                    if idcs_to_update is not None:
                        update_go_than = idcs_to_update.min()
                        for bead2update in self._incomplete_beads:
                            if bead2update is bead:
                                continue
                            bead2update._all_atom_idcs[bead2update._all_atom_idcs >= update_go_than] -= 1
                            # atomid2update_mask = np.isin(bead2update._all_atom_idcs, idcs_to_update)
                            # if sum(atomid2update_mask) > 0:
                            #     for atomid2update in bead2update._all_atom_idcs[atomid2update_mask]:
                            #         bead2update._all_atom_idcs[bead2update._all_atom_idcs >= atomid2update] -= 1
                    
                    self._check_bead_completeness(bead)
                
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb) # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]

                self.logger.error('An error occurred on line {} in statement {}'.format(line, text))
                raise
            except Exception as e:
                self.logger.warning(f"Missing {atom_idname} in mapping file")

        # Complete all beads. Missing atoms will be ignored.
        while len(self._incomplete_beads) > 0:
            self._complete_bead(self._incomplete_beads.pop())

        self._bead_idnames = np.array(bead_idnames)
        self._bead_resnames = np.array(bead_resnames)
        self._bead_names = np.array(bead_names)
        self._bead_resindices = np.array(bead_residcs)
        self._bead_resnums = np.array(bead_resnums)
        self._bead_segids = np.array(bead_segids)

        self._atom_idnames = np.empty(self._n_atoms, dtype="<U32")
        self._atom_resindices = np.empty(self._n_atoms, dtype=int)
        self._atom_resnums = np.empty(self._n_atoms, dtype=int)
        self._atom_segindices = np.empty(self._n_atoms, dtype="<U32")

        for bead in self._ordered_beads:
            self._atom_idnames[bead._all_atom_idcs] = bead._all_atom_idnames
            self._atom_resindices[bead._all_atom_idcs] = bead._all_atom_resindices
            self._atom_resnums[bead._all_atom_idcs] = bead._all_atom_resnums
            self._atom_segindices[bead._all_atom_idcs] = bead._all_atom_segids
        
        atom_resnames = []
        atom_names =    []
        for atom_idname in self._atom_idnames:
            atom_resname, atom_name = atom_idname.split(DataDict.STR_SEPARATOR)
            atom_resnames.append(atom_resname)
            atom_names.append(atom_name)
        
        self._atom_resnames = np.array(atom_resnames)
        self._atom_names = np.array(atom_names)
        
        self._compute_bead2atom_feats()
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        atom_forces = []
        bead_positions = []
        cell_dimensions = []

        all_atom_idcs = []
        build_all_atom_idcs = True

        self._initialize_extra_pos_impl()

        try:
            if len(self.trajectory) == 0:
                self.trajectory = [self.universe.trajectory.ts]

            for ts in self.trajectory:
                pos = np.empty((self._n_atoms, 3), dtype=float)
                forces = np.empty((self._n_atoms, 3), dtype=float)
                pos[...] = np.nan
                forces[...] = np.nan
                for bead in self._ordered_beads:
                    if build_all_atom_idcs:
                        all_atom_idcs.append(bead._all_atom_idcs)
                    pos[bead._all_atom_idcs] = bead.all_atom_positions
                    forces[bead._all_atom_idcs] = bead.all_atom_forces
                atom_positions.append(pos)
                atom_forces.append(forces)

                if ts.dimensions is not None:
                    cell_dimensions.append(cellpar_to_cell(ts.dimensions))
                
                not_nan_pos = np.nan_to_num(pos)
                bead_pos = np.sum(not_nan_pos[self._bead2atom_idcs] * self._bead2atom_weights[..., None], axis=1)
                bead_positions.append(bead_pos)
                self._update_extra_pos_impl(pos)

                build_all_atom_idcs = False
        
        except Exception as e:
            self.logger.error(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        
        self.all_atom_idcs = np.concatenate(all_atom_idcs)
        self._atom_positions =  np.stack(atom_positions, axis=0)
        self._atom_forces =  np.stack(atom_forces, axis=0)
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._cell = np.stack(cell_dimensions, axis=0) if len(cell_dimensions) > 0 else None
        self._store_extra_pos_impl()
        self._compute_extra_map_impl()
        return self

    def _compute_extra_map_impl(self):
        return

    def _initialize_extra_pos_impl(self):
        return

    def _update_extra_pos_impl(self, pos):
        return

    def _store_extra_pos_impl(self):
        return
    
    def _appearence_sort_idcs(self, atom_idname: str):
        def default_idname(atom_idname):
            return f"999_{atom_idname}"
        actual_idnames = self._atom2bead[atom_idname]
        appearence_idnames = [
            self._appearence_idnames.get(actual_idname, default_idname(atom_idname)) for actual_idname in actual_idnames
        ]
        return np.argsort(appearence_idnames)
    
    def _get_incomplete_bead_from_atom_idname(self, atom_idname: str) -> List[Bead]:
        idx = self._appearence_sort_idcs(atom_idname)
        bead_idnames = np.array(self._atom2bead[atom_idname])[idx]
        beads = []
        for bead_idname in bead_idnames:
            found = False
            for bead in self._incomplete_beads:
                if bead.idname.__eq__(bead_idname) and bead.is_missing_atom(atom_idname):
                    beads.append(bead)
                    found = True
            if not found:
                beads.append(self._create_bead(bead_idname))
        return beads

    def _update_bead(
        self,
        bead: Bead,
        atom_idname: str,
        atom: Optional[Atom]=None,
        atom_index: Optional[int]=None,
    ):
        bmas = self._bead_mapping_settings.get(bead.idname).get_bmas_by_atom_idname(atom_idname)
        if atom_index is not None:
            self._n_atoms -= 1
        return bead.update(atom_idname, bmas, atom=atom, atom_index=atom_index)
    
    def _create_bead(self, bead_idname: str, bead_class = Bead):
        bms = self._bead_mapping_settings.get(bead_idname)
        bead = bead_class(
            bms=bms,
            id=self._n_beads,
            idname=bead_idname,
            type=self._bead_types[bead_idname],
            atoms_offset=self._n_atoms,
            bead2atoms=copy.deepcopy(self._bead2atom[bead_idname]),
            weigth_based_on=self._weigth_based_on,
        )
        if bead_idname not in self._appearence_idnames:
            self._appearence_idnames[bead_idname] = "{:03d}".format(len(self._appearence_idnames.keys())) + f"_{bead_idname}"
        self._incomplete_beads.append(bead)
        self._ordered_beads.append(bead)
        self._n_beads += 1
        self._n_atoms += bead.n_all_atoms
        return bead
    
    def _complete_bead(self, bead: Bead):
        bead.complete()
        self._check_bead_completeness(bead)
    
    def _check_bead_completeness(self, bead: Bead):
        ''' Keep track of complete and incomplete beads, retaining the correct ordering.
        If a bead is complete, it is removed from the incomplete list and added to the complete list.
        '''
        if bead.is_complete and not (bead in self._complete_beads):
            self._complete_beads.append(bead)
            if bead in self._incomplete_beads:
                self._incomplete_beads.remove(bead)
            return True
        return False
    
    def _compute_bead2atom_feats(self):
        ### Initialize instance mapping ###
        self._bead2atom_idcs = -np.ones((self.num_beads, self.bead_all_size), dtype=int)
        self._bead2atom_weights = np.zeros((self.num_beads, self.bead_all_size), dtype=float)

        for i, bead in enumerate(self._ordered_beads):
            ### Build instance bead2atom_idcs and weights ###
            self._bead2atom_idcs[i, :bead.n_all_atoms] = bead._all_atom_idcs
            self._bead2atom_weights[i, :bead.n_all_atoms] = bead.all_atom_weights / bead.all_atom_weights.sum()

    def save(
        self,
        filename: Optional[str] = None,
        traj_format: Optional[str] = None,
    ):
        if filename is None:
            p = Path(self.config.get('input'))
            filename = self.config.get('output',  str(Path(p.parent, p.stem + '.CG' + p.suffix)))
            if os.path.isdir(filename):
                filename = os.path.join(filename, f"{random.getrandbits(128)}.pdb")
        
        if len(dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        dataset = self.dataset
        u = mda.Universe.empty(
            n_atoms =       dataset[DataDict.NUM_BEADS],
            n_residues =    dataset[DataDict.NUM_RESIDUES],
            n_segments =    len(np.unique(dataset[DataDict.BEAD_SEGIDS])),
            atom_resindex = dataset[DataDict.BEAD_RESIDCS],
            trajectory =    True, # necessary for adding coordinates
        )
        u.add_TopologyAttr('names',    dataset[DataDict.BEAD_NAMES])
        u.add_TopologyAttr('types',    dataset[DataDict.BEAD_TYPES])
        u.add_TopologyAttr('resnames', dataset[DataDict.RESNAMES])
        u.add_TopologyAttr('resid',    dataset[DataDict.RESIDCS])
        u.add_TopologyAttr('resnum',   dataset[DataDict.RESNUMBERS])
        u.add_TopologyAttr('chainID',  dataset[DataDict.BEAD_SEGIDS])
        u.load_new(np.nan_to_num(dataset.get(DataDict.BEAD_POSITION)), order='fac')
        box_dimension = dataset.get(DataDict.CELL, None)
        if box_dimension is not None:
            for ts in u.trajectory:
                ts.dimensions = box_dimension[ts.frame]
        selection = u.select_atoms('all')
        with mda.Writer(filename, n_atoms=u.atoms.n_atoms) as w:
            w.write(selection.atoms)

        filename_traj = None
        if len(u.trajectory) > 1:
            if traj_format is None:
                traj_format = self.config.get('outputtraj', 'xtc')
            filename_traj = str(Path(filename).with_suffix('.' + traj_format))
            with mda.Writer(filename_traj, n_atoms=u.atoms.n_atoms) as w_traj:
                for ts in u.trajectory:  # Skip the first frame as it's already saved
                    w_traj.write(selection.atoms)
        
        self.logger.info(f"Coarse-grained structure saved as {filename}")
        return filename, filename_traj
    
    def save_atomistic(
        self,
        filename: Optional[str] = None,
        traj_format: Optional[str] = None,
    ):
        if filename is None:
            p = Path(self.config.get('input'))
            filename = self.config.get('output',  str(Path(p.parent, p.stem + '.AA' + p.suffix)))
        
        if len(dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        dataset = self.dataset
        u = mda.Universe.empty(
            n_atoms =       dataset[DataDict.NUM_ATOMS],
            n_residues =    dataset[DataDict.NUM_RESIDUES],
            n_segments =    len(np.unique(dataset[DataDict.ATOM_SEGIDS])),
            atom_resindex = dataset[DataDict.ATOM_RESIDCS],
            trajectory =    True, # necessary for adding coordinates
        )
        u.add_TopologyAttr('names',    dataset[DataDict.ATOM_NAMES])
        u.add_TopologyAttr('types',    dataset[DataDict.ATOM_TYPES])
        u.add_TopologyAttr('resnames', dataset[DataDict.RESNAMES])
        u.add_TopologyAttr('resid',    dataset[DataDict.RESIDCS])
        u.add_TopologyAttr('resnum',   dataset[DataDict.RESNUMBERS])
        u.add_TopologyAttr('chainID',  dataset[DataDict.ATOM_SEGIDS])
        u.load_new(np.nan_to_num(dataset.get(DataDict.ATOM_POSITION)), order='fac')
        box_dimension = dataset.get(DataDict.CELL, None)
        if box_dimension is not None:
            for ts in u.trajectory:
                ts.dimensions = box_dimension[ts.frame]
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
        
        self.logger.info(f"Atomistic structure saved as {filename}")
    
    def save_npz(
        self,
        filename: Optional[str] = None,
        from_pos_unit: str = 'nm',
        to_pos_unit: str = 'nm',
        from_force_unit: str = 'kJ/(mol*nm)',
        to_force_unit: str = 'kJ/(mol*nm)',
    ):
        """Save a npz dataset with all atomistic and mapped CG data.

        Args:
            filename (str): name of the output file. Must be a .npz file
            pos_unit (str): unit of measure of the atoms/beads positions. Options are [
                'Angstrom', 'A', 'angstrom', 'Å', 'nm', 'nanometer', 'pm', 'picometer', 'fm', 'femtometer'
            ]. Default is 'Angstrom'.
            force_unit (str): unit of measure of the atoms/beads forces. Options are [
                'kJ/(mol*Angstrom)', 'kJ/(mol*A)', 'kJ/(mol*Å)', 'kJ/(mol*nm)', 'Newton', 'N', 'J/m', 'kcal/(mol*Angstrom)'
            ].  Default is 'kJ/(mol*Angstrom)'.
    """
        
        dataset = self.dataset

        for key in [DataDict.BEAD_POSITION, DataDict.ATOM_POSITION]:
            dataset[key] = mda.units.convert(dataset[key], from_pos_unit, to_pos_unit)
        
        for key in [DataDict.BEAD_FORCES, DataDict.ATOM_FORCES]:
            if key in dataset:
                dataset[key] = mda.units.convert(dataset[key], from_force_unit, to_force_unit)
        
        if DataDict.CELL in dataset:
            cell = dataset[DataDict.CELL]
            cell[:, :3] = mda.units.convert(cell[:, :3], from_pos_unit, to_pos_unit)
        
        if filename is None:
            p = Path(self.config.get('input'))
            filename = str(Path(p.parent, p.stem + '.data' + '.npz'))
        else:
            p = Path(filename)
            if p.suffix != '.npz':
                filename = filename + '.npz'
        
        if len(dirname(filename)) > 0:
            os.makedirs(dirname(filename), exist_ok=True)
        np.savez(filename, **dataset)
        self.logger.info(f"npz dataset saved as {filename}")