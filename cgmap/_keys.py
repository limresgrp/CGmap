import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# Generic keys

STR_SEPARATOR: str = '_'

# ------------- DataDict Keys ----------------- #

NUM_RESIDUES: Final[str] = "num_residues"
RESNAMES: Final[str] = "resnames"
RESIDCS: Final[str] = "residcs"
RESNUMBERS: Final[str] = "resnumbers"

# --------------------------------------------- #

NUM_ATOMS: Final[str] = "num_atoms"
ATOM_POSITION: Final[str] = "atom_pos"
ATOM_RESNAMES: Final[str] = "atom_resnames"
ATOM_NAMES: Final[str] = "atom_names"
ATOM_TYPES: Final[str] = "atom_types"
ATOM_RESIDCS: Final[str] = "atom_residcs"
ATOM_RESNUMBERS: Final[str] = "atom_resnumbers"
ATOM_SEGIDS: Final[str] = "ATOM_SEGIDS"
ATOM_FORCES: Final[str] = "atom_forces"

# --------------------------------------------- #

NUM_BEADS: Final[str] = "num_beads"
BEAD_POSITION: Final[str] = "bead_pos"
BEAD_IDNAMES: Final[str] = "bead_idnames"
BEAD_RESNAMES: Final[str] = "bead_resnames"
BEAD_NAMES: Final[str] = "bead_names"
BEAD_TYPES: Final[str] = "bead_types"
BEAD_RESIDCS: Final[str] = "bead_residcs"
BEAD_RESNUMBERS: Final[str] = "bead_resnumbers"
BEAD_SEGIDS: Final[str] = "bead_segids"
BEAD_FORCES: Final[str] = "bead_forces"

# --------------------------------------------- #

BOND_IDCS:  Final[str] = "bond_idcs"
ANGLE_IDCS:  Final[str] = "angle_idcs"
DIHEDRAL_IDCS: Final[str] = "dihedral_idcs"
CELL: Final[str] = "cell"
PBC: Final[str] = "pbc"

BEAD2ATOM_IDCS: Final[str] = "bead2atom_idcs"
BEAD2ATOM_WEIGHTS: Final[str] = "bead2atom_weights"

