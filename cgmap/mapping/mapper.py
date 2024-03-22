import copy
import logging
import re
import os
import sys
import yaml
import traceback
import numpy as np
import MDAnalysis as mda

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from MDAnalysis.core.groups import Atom

from .bead import BeadMappingSettings, BeadMappingAtomSettings, Bead
from cgmap.utils import DataDict
from cgmap.utils.atomType import get_type_from_name


class Mapper():

    u: mda.Universe = None
    bead_types_filename: str = "bead_types.yaml"

    _weigth_based_on: str = 'mass'
    _atom2bead: dict[str, str] = {}
    _bead2atom: dict[str, List[str]] = {}
    _bead_types: dict[str, int] = {}
    _bead_mapping_settings: dict[str, BeadMappingSettings] = {}
    _bead_cm: dict[str, str] = {}

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

    _bead2atom_idcs: np.ndarray = None
    _bead2atom_weights: np.ndarray = None
    _bead2atom_mask: np.ndarray = None
    
    _cell: np.ndarray = None
    _pbc: np.ndarray = np.array([True, True, True])

    

    # Properties of mapping
    
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
    
    @property
    def bead2atom_mask(self):
        return self._bead2atom_mask

    # Properties of mapped instance
    
    @property
    def bead2atom_idcs(self):
        if self._bead2atom_idcs is None:
            return None
        return np.ma.masked_array(self._bead2atom_idcs, mask=~self.bead2atom_mask)
    
    @property
    def bead2atom_weights(self):
        if self._bead2atom_weights is None:
            return None
        return np.ma.masked_array(self._bead2atom_weights, mask=~self.bead2atom_mask)
    
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
            DataDict.ATOM_FORCES:     self._atom_forces,

            DataDict.NUM_BEADS:       self.num_beads,
            DataDict.BEAD_POSITION:   self._bead_positions,
            DataDict.BEAD_IDNAMES:    self._bead_idnames,
            DataDict.BEAD_RESNAMES:   self._bead_resnames,
            DataDict.BEAD_NAMES:      self._bead_names,
            DataDict.BEAD_TYPES:      self.bead_types,
            DataDict.BEAD_RESIDCS:    self.bead_resindices,
            DataDict.BEAD_RESNUMBERS: self._bead_resnums,
            DataDict.BEAD_SEGIDS:     self._bead_segids,

            DataDict.CELL:            self._cell,
            DataDict.PBC:             self._pbc,

            DataDict.BEAD2ATOM_IDCS:      self._bead2atom_idcs,
            DataDict.BEAD2ATOM_WEIGHTS:   self._bead2atom_weights,
        }.items() if v is not None}

    def __init__(self, config: Dict, logger = None) -> None:
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger()
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
            self._root = mapping_folder
        else:
            self._root = os.path.join(os.path.dirname(__file__), '..', 'data', mapping_folder)
        self._weigth_based_on = config.get("weigth_based_on", "mass")

        # Iterate configuration files and load all mappings
        self._clear_mappings()
        self._load_mappings()

    def _clear_mappings(self):
        self._atom2bead.clear()
        self._bead2atom.clear()
        self._bead_types.clear()
        self._bead_mapping_settings.clear()
        
        self._bead2atom_mask: np.ndarray = None
        self._max_bead_reconstructed_atoms: int = 0
        self._max_bead_all_atoms: int = 0

    def _load_mappings(self):
        # Load bead types file, if existent.
        # It contains the bead type to identify each bead inside the NN
        # Different beads could have the same bead type (for example, all backbone beads could use the same bead type)
        bead_types_filename = os.path.join(self._root, self.bead_types_filename)
        if os.path.isfile(bead_types_filename):
            bead_types_conf: dict = yaml.safe_load(Path(bead_types_filename).read_text())
        else:
            bead_types_conf: dict = dict()
        
        # Iterate mapping files -> 1 mapping file = 1 residue mapping
        for filename in os.listdir(self._root):
            if filename == self.bead_types_filename:
                continue
            
            conf: dict = OrderedDict(yaml.safe_load(Path(os.path.join(self._root, filename)).read_text()))
            mol = conf.get("molecule")
            
            _conf_bead2atom = OrderedDict({})

            for atom, bead_settings_str in conf.get("atoms").items():
                
                all_bead_settings = bead_settings_str.split()
                bead_names = all_bead_settings[0].split(',')

                for i, bn in enumerate(bead_names):
                    bead_idname = DataDict.STR_SEPARATOR.join([mol, bn])
                    atom_idname = DataDict.STR_SEPARATOR.join([mol, atom])
                    bead_settings = [x.split(',')[i] for x in all_bead_settings[1:]]
                    
                    atom2bead_list = self._atom2bead.get(atom_idname, [])
                    atom2bead_list.append(bead_idname)
                    self._atom2bead[atom_idname] = atom2bead_list
                    bms = self._bead_mapping_settings.get(bead_idname, BeadMappingSettings(bead_idname))
                    bmas = bms.get_bmas_by_atom_idname(atom_idname)
                    if bmas is None:
                        bmas = BeadMappingAtomSettings(bead_settings, bead_idname, atom_idname, num_shared_beads=len(bead_names))
                        bms.add_atom_settings(bmas)
                        self._bead_mapping_settings[bead_idname] = bms
                    
                    bead2atom: List[str] = _conf_bead2atom.get(bead_idname, [])
                    if len(bead2atom) == 0 and bead_idname not in self._bead_types:
                        bead_type = bead_types_conf.get(bead_idname, max(bead_types_conf.values(), default=-1)+1)
                        bead_types_conf[bead_idname] = bead_type
                        self._bead_types[bead_idname] = bead_type
                    assert atom_idname not in bead2atom, f"{atom_idname} is already present in bead {bead_idname}. Duplicate mapping"
                    bead2atom.append(atom_idname)
                    _conf_bead2atom[bead_idname] = bead2atom
            
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

        ### Compute global mask and maximum bead size ###
        self._bead2atom_mask = np.zeros((self.n_bead_types, self.bead_reconstructed_size), dtype=bool)
                
        for bead_idname, b2a in self._bead2atom.items():
            self._bead2atom_mask[self._bead_types[bead_idname], :len(b2a[0])] = True

    def _load_extra_mappings(self, bead_splits, atom_name, bead_idname):
        return
    
    def _clear_map(self):
        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._ordered_beads.clear()
        self._n_beads = 0
        self._n_atoms = 0

    def mapcg(self):
        self._clear_map()
        
        self.trajslice = self.config.get("trajslice", None)
        self.u = mda.Universe(self.config.get("input"), *self.config.get("inputtraj", []), **self.config.get("extrakwargs", {}))

        selection = self.config.get("selection", "all")
        self.sel = self.u.select_atoms(selection)
        
        return self.map_impl_cg()
    
    def map(self):
        self._clear_map()
        
        self.trajslice = self.config.get("trajslice", None)
        self.u = mda.Universe(self.config.get("input"), *self.config.get("inputtraj", []), **self.config.get("extrakwargs", {}))

        selection = self.config.get("selection", "all")
        self.sel = self.u.select_atoms(selection)
        
        return self.map_impl()
    
    def map_impl_cg(
        self,
        conf_id: int = 0 # Which configuration to select for naming atoms
    ):
        self._bead_resnames =   self.sel.atoms.resnames
        self._bead_names =      self.sel.atoms.names
        self._bead_idnames = np.array([
            f"{resname}{DataDict.STR_SEPARATOR}{bead}"
            for resname, bead in zip(self._bead_resnames, self._bead_names)
        ])
        self._bead_resindices = self.sel.atoms.resindices
        self._bead_resnums =    self.sel.atoms.resnums
        self._bead_segids =     self.sel.atoms.segids

        bead_positions = []
        cell_dimensions = []
        
        try:
            traj = self.u.trajectory if self.trajslice is None else self.u.trajectory[self.trajslice]
            for ts in traj:
                bead_pos = self.sel.positions
                if ts.dimensions is not None:
                    cell_dimensions.append(ts.dimensions)
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

        self._atom_idnames =    np.array(atom_idnames)
        self._atom_resnames =   np.array(atom_resnames)
        self._atom_names =      np.array(atom_names)
        self._atom_resindices = np.array(atom_resindices)
        self._atom_resnums =    np.array(atom_resnums)
        self._atom_segindices = np.array(atom_segindices)

        batch, _, xyz = self._bead_positions.shape
        self._atom_positions =  np.zeros((batch, len(self._atom_names), xyz), dtype=self._bead_positions.dtype)

        self.compute_bead2atom_feats()
        self.compute_extra_map_impl()
            
    def map_impl(self):
        
        bead_idnames =  []
        bead_resnames = []
        bead_names =    []
        bead_residcs =  []
        bead_resnums =  []
        bead_segids =   []

        last_resindex = -1
        for atom in self.sel.atoms:
            try:
                atom_idname = DataDict.STR_SEPARATOR.join([atom.resname, re.sub(r'^(\d+)\s*(.+)', r'\2\1', atom.name)])

                # This check is necessary to complete beads on residue change.
                # This allows having beads with incomplete atoms that do not remain pending
                if atom.resindex > last_resindex:
                    for bead in self._incomplete_beads:
                        self._complete_bead(bead)
                        self._check_bead_completeness(bead)
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
                        for bead2update in self._incomplete_beads:
                            if bead2update is bead:
                                continue
                            atomid2update_mask = np.isin(bead2update._all_atom_idcs, idcs_to_update)
                            if sum(atomid2update_mask) > 0:
                                for atomid2update in bead2update._all_atom_idcs[atomid2update_mask]:
                                    bead2update._all_atom_idcs[bead2update._all_atom_idcs >= atomid2update] -= 1
                    
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
        for bead in self._incomplete_beads:
            self._complete_bead(bead)
            self._check_bead_completeness(bead)

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
        
        self.compute_bead2atom_feats()
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        atom_forces = []
        bead_positions = []
        cell_dimensions = []

        all_atom_idcs = []
        build_all_atom_idcs = True

        self.initialize_extra_pos_impl()

        try:
            traj = self.u.trajectory if self.trajslice is None else self.u.trajectory[self.trajslice]
            for ts in traj:

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
                    cell_dimensions.append(ts.dimensions)
                
                not_nan_pos = np.nan_to_num(pos)
                bead_pos = np.sum(not_nan_pos[self._bead2atom_idcs] * self._bead2atom_weights[..., None], axis=1)
                bead_positions.append(bead_pos)
                self.update_extra_pos_impl(pos)

                build_all_atom_idcs = False
        
        except Exception as e:
            self.logger.error(f"Error rading trajectory: {e}. Skipping missing trajectory frames.")
        
        self.all_atom_idcs = np.concatenate(all_atom_idcs)
        self._atom_positions =  np.stack(atom_positions, axis=0)
        self._atom_forces =  np.stack(atom_forces, axis=0)
        self._bead_positions =  np.stack(bead_positions, axis=0)
        self._cell = np.stack(cell_dimensions, axis=0) if len(cell_dimensions) > 0 else None
        self.store_extra_pos_impl()
        self.compute_extra_map_impl()

    def compute_extra_map_impl(self):
        return

    def initialize_extra_pos_impl(self):
        return

    def update_extra_pos_impl(self, pos):
        return

    def store_extra_pos_impl(self):
        return
    
    def _get_incomplete_bead_from_atom_idname(self, atom_idname: str) -> List[Bead]:
        bead_idnames = np.unique(self._atom2bead[atom_idname])
        beads = []
        for bead_idname in bead_idnames:
            found = False
            for bead in self._incomplete_beads:
                if bead.idname.__eq__(bead_idname) and bead.is_missing_atom(atom_idname):
                    beads.append(bead)
                    found = True
                    break
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
    
    def _create_bead(self, bead_idname: str):
        bms = self._bead_mapping_settings.get(bead_idname)
        bead = Bead(
            bms=bms,
            id=self._n_beads,
            idname=bead_idname,
            type=self._bead_types[bead_idname],
            atoms_offset=self._n_atoms,
            bead2atoms=copy.deepcopy(self._bead2atom[bead_idname]),
            weigth_based_on=self._weigth_based_on,
        )
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
    
    def compute_bead2atom_feats(self):
        ### Initialize instance mapping ###
        self._bead2atom_idcs = -np.ones((self.num_beads, self.bead_all_size), dtype=int)
        self._bead2atom_weights = np.zeros((self.num_beads, self.bead_all_size), dtype=float)

        for i, bead in enumerate(self._ordered_beads):
            ### Build instance bead2atom_idcs and weights ###
            self._bead2atom_idcs[i, :bead.n_all_atoms] = bead._all_atom_idcs
            self._bead2atom_weights[i, :bead.n_all_atoms] = bead.all_atom_weights / bead.all_atom_weights.sum()

    def save(self, filename: Optional[str] = None, traj_format: Optional[str] = None):

        if filename is None:
            p = Path(self.config.get('input'))
            filename = self.config.get('output',  str(Path(p.parent, p.stem + '_CG' + p.suffix)))
        
        if len(os.path.dirname(filename)) > 0:
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
        u.dimensions = dataset.get(DataDict.CELL, None)
        sel = u.select_atoms('all')
        with mda.Writer(filename, n_atoms=u.atoms.n_atoms) as w:
            w.write(sel.atoms)

        if len(u.trajectory) > 1:
            if traj_format is None:
                traj_format = self.config.get('outputtraj', 'xtc')
            filename_traj = str(Path(filename).with_suffix('.' + traj_format))
            with mda.Writer(filename_traj, n_atoms=u.atoms.n_atoms) as w_traj:
                for ts in u.trajectory:  # Skip the first frame as it's already saved
                    w_traj.write(sel.atoms)