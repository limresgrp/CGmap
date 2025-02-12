import logging
import re
import itertools
import numpy as np

from typing import Optional, List, Union
from MDAnalysis.core.groups import Atom
from cgmap.utils import DataDict
from cgmap.utils.atomType import get_mass_from_name


class BeadMappingAtomSettings:

    def __init__(
        self,
        bead_settings: List[str],
        bead_name: str,
        atom_idnames: List[str],
        num_shared_beads: int
    ) -> None:
        self.bead_name: str = bead_name
        self.atom_idnames: List[str] = atom_idnames
        self.atom_names = [atom_idname.split(DataDict.STR_SEPARATOR)[1] for atom_idname in atom_idnames]
        self.atom_resname = atom_idnames[0].split(DataDict.STR_SEPARATOR)[0]

        self._num_shared_beads: int = num_shared_beads

        self._contributes_to_cm: bool = True
        self._is_cm: bool = False
        self._has_cm: bool = False
        self._mass: float = get_mass_from_name(self.atom_names[0])
        self._relative_weight: float = 1.
        self._relative_weight_set: bool = False

        self._has_to_be_reconstructed: bool = True

        for setting in bead_settings:
            try:
                if setting == "!":
                    self.exclude_from_cm()
                    continue
                if setting == "CM":
                    self.set_is_cm()
                    continue
                
                weight_pattern = '(\d)/(\d)'
                result = re.search(weight_pattern, setting)
                if result is not None:
                    groups = result.groups()
                    assert len(groups) == 2
                    self.set_relative_weight(float(int(groups[0])/int(groups[1])))
                    continue

                weight_pattern = '(?<![A-Za-z])(\d+(?:\.\d+)?)(?![A-Za-z])'
                result = re.search(weight_pattern, setting)
                if result is not None:
                    groups = result.groups()
                    assert len(groups) == 1
                    self.set_relative_weight(float(groups[0]))
                    continue
            except ValueError as e:
                logging.error(f"Error while parsing mapping for bead {self.bead_name}. Incorrect mapping setting: {setting}")
                raise e
        
        if not self._relative_weight_set:
            self.set_relative_weight(self.relative_weight / self._num_shared_beads)

    @property
    def contributes_to_cm(self):
        return self._contributes_to_cm

    @property
    def is_cm(self):
        return self._is_cm

    @property
    def has_cm(self):
        return self._has_cm
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def has_to_be_reconstructed(self):
        return self._has_to_be_reconstructed
    
    @property
    def relative_weight(self):
        return self._relative_weight
    
    def exclude_from_cm(self):
        self._contributes_to_cm = False

    def set_is_cm(self, is_cm: bool = True):
        self._is_cm = is_cm
        self.set_has_cm(is_cm)
    
    def set_has_cm(self, has_cm: bool = True):
        self._has_cm = has_cm
    
    def set_has_to_be_reconstructed(self, has_to_be_reconstructed: bool = True):
        self._has_to_be_reconstructed = has_to_be_reconstructed
    
    def set_relative_weight(self, weight: float):
        self._relative_weight_set = True
        self._relative_weight = weight


class BeadMappingSettings:

    def __init__(self, bead_idname) -> None:
        self._bead_idname: str = bead_idname
        self._is_complete = False
        self._bead_reconstructed_size = 0
        self._bead_all_size = 0

        self._atom_settings: List[BeadMappingAtomSettings] = []
    
    @property
    def shared_atoms(self):
        return [_as.atom_idnames for _as in self._atom_settings if _as._has_to_be_reconstructed and _as._num_shared_beads > 1]

    @property
    def bead_reconstructed_size(self):
        assert self._is_complete
        return self._bead_reconstructed_size
    
    @property
    def bead_all_size(self):
        assert self._is_complete
        return self._bead_all_size

    def add_atom_settings(self, bmas: BeadMappingAtomSettings):
        self._atom_settings.append(bmas)
        self.update_atom_settings(bmas)
    
    def update_atom_settings(self, bmas: BeadMappingAtomSettings):
        has_cm = bmas.is_cm
        for saved_bmas in self._atom_settings:
            if saved_bmas.has_cm:
                has_cm = True
            if has_cm:    
                saved_bmas.set_has_cm()
    
    def complete(self):
        self.update_relative_weights()
        if not self._is_complete:
            self._bead_reconstructed_size = sum([atom_setting._has_to_be_reconstructed for atom_setting in self._atom_settings])
            self._bead_all_size = len(self._atom_settings)
            self._is_complete = True

    def update_relative_weights(self):
        total_relative_weight = sum([_as.relative_weight for _as in self._atom_settings])
        for bmas in self._atom_settings:
            bmas.set_relative_weight(bmas.relative_weight / total_relative_weight)

    def get_bmas_by_atom_idname(self, atom_idnames: Union[str, List[str]]):
        for bmas in self._atom_settings:
            if isinstance(atom_idnames, str):
                if atom_idnames in bmas.atom_idnames:
                    return bmas
            elif all([atom_idname in bmas.atom_idnames for atom_idname in atom_idnames]):
                return bmas
        return None


class Bead:

    def __init__(
            self,
            bms: BeadMappingSettings,
            id: int,
            idname: str,
            type: int,
            atoms_offset: int,
            bead2atoms: List[List[str]], # Could have multiple configuration files with different atom namings
            weigth_based_on: str,
            resindex: int = 0,
            resnum: int = 0,
            segid: str = 'A',
    ) -> None:
        self.bms = bms
        self.id = id
        self.idname = idname
        self.resname, self.name = self.idname.split(DataDict.STR_SEPARATOR)
        self.type = type # Used by the NN
        self.resindex = resindex
        self.resnum = resnum
        self.segid = segid
        self.weigth_based_on = weigth_based_on

        self._n_found_atoms = 0
        self._is_complete: bool = False
        self._is_newly_created: bool = True
        
        assert self.n_all_atoms == len(bead2atoms[0])

        def numpy(list_of_tuples: List[tuple]):
            max_dimension = max(len(t) for t in list_of_tuples)
            return np.array([t + ('',) * (max_dimension - len(t)) for t in list_of_tuples])

        self._config_ordered_atom_idnames: List[np.ndarray] = [numpy(b2a) for b2a in bead2atoms]
        
        self._eligible_atom_idnames: set[str] = {
            item
            for tuple_ in itertools.chain(*bead2atoms)
            for item in tuple_
        }
        self._alternative_name_index = np.zeros((self.n_all_atoms, ), dtype=np.int16)

        self._all_atoms:        List[Atom] = []
        self._all_atom_idnames: List[str]  = []

        self._all_atom_idcs = np.arange(atoms_offset, atoms_offset + self.n_all_atoms)
        self._all_atom_weights = np.zeros((self.n_all_atoms, ), dtype=np.float32)
        
        self._reconstructed_atom_idnames: List[str]      = []
        self._reconstructed_conf_ordered_idcs: List[int] = []

        self._reconstructed_atom_idcs = np.zeros((self.n_reconstructed_atoms,), dtype=np.int16)
        self._reconstructed_atom_weights = np.zeros((self.n_reconstructed_atoms,), dtype=np.float32)

        self.missing_atoms_idcs = []
    
    @property
    def n_all_atoms(self):
        return self.bms.bead_all_size
    
    @property
    def n_reconstructed_atoms(self):
        return self.bms.bead_reconstructed_size
    
    @property
    def is_newly_created(self):
        return self._is_newly_created
    
    @property
    def is_complete(self):
        return self._is_complete
    
    @property
    def all_atom_positions(self):
        if self.is_complete and len(self._all_atoms) > 0:
            atoms_positions = np.empty((self.n_all_atoms, 3), dtype=np.float32)
            atoms_positions[...] = np.nan
            try:
                actual_atoms_mask = np.ones((self.n_all_atoms,), dtype=bool)
                if len(self.missing_atoms_idcs) > 0:
                    actual_atoms_mask[self.missing_atoms_idcs] = False
                atoms_positions[actual_atoms_mask] = np.stack([atom.position for atom in self._all_atoms], axis=0)
                return atoms_positions
            except:
                return atoms_positions
        return None

    @property
    def all_atom_forces(self):
        if self.is_complete and len(self._all_atoms) > 0:
            try:
                atoms_forces = np.empty((self.n_all_atoms, 3), dtype=np.float32)
                atoms_forces[...] = np.nan
                actual_atoms_mask = np.ones((self.n_all_atoms,), dtype=bool)
                if len(self.missing_atoms_idcs) > 0:
                    actual_atoms_mask[self.missing_atoms_idcs] = False
                atoms_forces[actual_atoms_mask] = np.stack([atom.force for atom in self._all_atoms], axis=0)
                return atoms_forces
            except:
                return None
        return None
    
    @property
    def all_atom_weights(self):
        all_atom_weights = np.zeros_like(self._all_atom_weights)
        actual_atoms_mask = np.ones((self.n_all_atoms,), dtype=bool)
        if len(self.missing_atoms_idcs) > 0:
            actual_atoms_mask[self.missing_atoms_idcs] = False
        all_atom_weights[actual_atoms_mask] = self._all_atom_weights[actual_atoms_mask]
        return all_atom_weights
    
    @property
    def _all_atom_resindices(self):
        return np.array([self.resindex] * self.n_all_atoms)
    
    @property
    def _all_atom_resnums(self):
        return np.array([self.resnum] * self.n_all_atoms)
    
    @property
    def _all_atom_segids(self):
        return np.array([self.segid] * self.n_all_atoms)
    
    def is_missing_atom(self, atom_idname: str):
        assert not self.is_complete, "Can only call this method before the bead is complete."
        return atom_idname in self._eligible_atom_idnames and ~np.isin(atom_idname, self._all_atom_idnames)
    
    def scale_bead_idcs(self, atom_index_offset: int):
        self._all_atom_idcs -= atom_index_offset

    def update(
        self,
        atom_idname: str,
        bmas: BeadMappingAtomSettings,
        atom: Optional[Atom] = None,
        atom_index: Optional[int] = None,
    ):
        if self._is_newly_created and atom is not None:
            self.resindex = atom.resindex
            self.resnum   = atom.resnum
            self.segid    = atom.segid
        self._is_newly_created = False
        assert atom_idname in self._eligible_atom_idnames, f"Trying to update bead {self.name} with atom {atom_idname} that does not belong to it."

        conf_ordered_index = None
        updated_config_ordered_atom_idnames = []
        for coaidnames in self._config_ordered_atom_idnames:
            coai_index = np.argwhere(coaidnames == atom_idname)
            if len(coai_index) > 0:
                updated_config_ordered_atom_idnames.append(coaidnames)
                if conf_ordered_index is None:
                    for coaid in coai_index:
                        conf_ordered_index = coaid[0]
                        self._alternative_name_index[conf_ordered_index] = coaid[1]
                        break
            else:
                pass
        if conf_ordered_index is None:
            raise Exception(f"Atom with idname {atom_idname} not found in mapping configuration files for bead {self.idname}")
        self._config_ordered_atom_idnames = updated_config_ordered_atom_idnames
        
        self._n_found_atoms += 1

        if atom is not None:
            self._all_atoms.append(atom)
        self._all_atom_idnames.append(atom_idname)

        # All atoms without the '!' flag contribute to the bead position
        # Those with the '!' flag appear with weight 0
        weight = 0.
        if bmas.has_cm:
            weight = 1. * bmas.is_cm
        elif bmas.contributes_to_cm:
            if self.weigth_based_on == "same":
                weight = 1.
            elif self.weigth_based_on == "mass":
                weight = bmas.mass
            else:
                raise Exception(f"{self.weigth_based_on} is not a valid value for 'weigth_based_on'. Use either 'mass' or 'same'")
        weight *= bmas.relative_weight
        self._all_atom_weights[conf_ordered_index] = weight

        idcs_to_update = None
        if atom_index is not None:
            # If atom index is not None, it means that this atom is shared with another bead,
            # and its idname, position and other properties have already been stored.
            # Thus, on this bead we adjust the index to point to the previously saved one (atom_index)
            # and we scale atom indices that are greater to avoid having gaps in the indexing.
            shared_atom_index = self._all_atom_idcs[conf_ordered_index]
            self._all_atom_idcs[conf_ordered_index] = atom_index
            idcs_to_update = np.copy(self._all_atom_idcs[self._all_atom_idcs > shared_atom_index])
            self._all_atom_idcs[self._all_atom_idcs > shared_atom_index] -= 1
        
        if bmas.has_to_be_reconstructed:
            self._reconstructed_conf_ordered_idcs.append(conf_ordered_index)
            self._reconstructed_atom_idnames.append(atom_idname)

        if self._n_found_atoms == self.n_all_atoms:
            self.complete()
        
        return self._all_atom_idcs[conf_ordered_index], idcs_to_update
    
    def complete(self):
        if self.is_complete:
            return
        # We need to reorder the atoms in the bead according to the order of the mapping.
        # This is necessary since when doing backmapping directly from CG we reconstruct
        # atom positions hierarchily following the order of the mapping config, while when
        # reading an atomistic structure the order of the atoms is that of appeareance in the pdb.
        # We need consistency among atomistic and CG, otherwise the NN trained on the order of the
        # atomistic pdbs may swap the prediction order in the CG if atoms appear in different order in the config
        # w.r.t. the order in the pdbs used for training.
        self._all_atom_idnames = np.array(self._all_atom_idnames)
        self._reconstructed_atom_idnames = np.array(self._reconstructed_atom_idnames)
        self._reconstructed_conf_ordered_idcs = np.array(self._reconstructed_conf_ordered_idcs, dtype=int)

        # ------------------------------------------------------------------------------------------------------------------ #

        config_ordered_all_atom_idnames, all_atom_idnames_sorted_idcs = self.sort_atom_idnames(self._all_atom_idnames)
        all_sorting_filter = all_atom_idnames_sorted_idcs[np.searchsorted(
            self._all_atom_idnames[all_atom_idnames_sorted_idcs], config_ordered_all_atom_idnames
        )]
        
        if len(self._all_atoms) > 0:
            self._all_atoms = np.array(self._all_atoms)[all_sorting_filter]
        self._all_atom_idnames = np.array(self._all_atom_idnames)[all_sorting_filter]

        config_ordered_reconstructed_atom_idnames, reconstructed_atom_idnames_sorted_idcs = self.sort_atom_idnames(self._reconstructed_atom_idnames)
        reconstructed_sorting_filter = reconstructed_atom_idnames_sorted_idcs[np.searchsorted(
            self._reconstructed_atom_idnames[reconstructed_atom_idnames_sorted_idcs],
            config_ordered_reconstructed_atom_idnames
        )]

        self._reconstructed_atom_idnames = self._reconstructed_atom_idnames[reconstructed_sorting_filter]
        self._reconstructed_conf_ordered_idcs = self._reconstructed_conf_ordered_idcs[reconstructed_sorting_filter]
        self._reconstructed_atom_idcs = self._all_atom_idcs[self._reconstructed_conf_ordered_idcs]
        self._reconstructed_atom_weights = self._all_atom_weights[self._reconstructed_conf_ordered_idcs]

        config_ordered_atom_idnames = self._config_ordered_atom_idnames[0]
        oaoffset = 0
        for oaindex, oaidname in enumerate(config_ordered_atom_idnames):
            if (oaindex > len(self._all_atom_idnames)+oaoffset-1) or ~(np.isin(self._all_atom_idnames[oaindex - oaoffset], oaidname)):
                self.missing_atoms_idcs.append(oaindex)
                oaoffset += 1
        self.missing_atoms_idcs = np.array(self.missing_atoms_idcs)
        self._all_atom_idnames = config_ordered_atom_idnames[np.arange(len(config_ordered_atom_idnames)), self._alternative_name_index]

        self._is_complete = True

    def sort_atom_idnames(self, atom_idnames):
        coan_max_len = 0
        config_ordered_atom_idnames = np.array([], dtype=int)

        def build_coan(coa, atom_idnames):
            coan_list = [x[np.isin(x, atom_idnames)] for x in coa if np.any(np.isin(x, atom_idnames))]
            coan_selected = np.concatenate([x for x in coan_list if len(x) == 1])

            while len(coan_selected) < len(atom_idnames):
                coan_list_new = []
                for coan_elem in coan_list:
                    if len(coan_elem) == 1:
                        coan_list_new.append(coan_elem)
                    else:
                        coan_list_new.append(coan_elem[~np.isin(coan_elem, coan_selected)])
                coan_list = coan_list_new
                coan_selected = np.concatenate([x for x in coan_list if len(x) == 1])
            return coan_selected

        for coa in self._config_ordered_atom_idnames:
            coan = build_coan(coa, atom_idnames)
            if len(coan) > coan_max_len:
                coan_max_len = len(coan)
                config_ordered_atom_idnames = coan
        atom_idnames_sorted_idcs = np.argsort(atom_idnames)
        return config_ordered_atom_idnames, atom_idnames_sorted_idcs