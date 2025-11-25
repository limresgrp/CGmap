# CGmap

**CGmap** is a flexible and powerful tool for mapping atomistic simulations to various coarse-grained (CG) representations.  
It allows users to define their own custom CG mappings and process everything from single structures to entire trajectories.

---

## 📝 Overview

This repository provides a Python-based tool to map atomistic structures to coarse-grained (CG) representations.  

The core philosophy of CGmap is **flexibility**: users can easily define their own CG mappings via simple YAML files.  

The output is compatible with tools like [HEroBM](https://github.com/limresgrp/HEroBM), enabling a full cycle of **mapping (atomistic → CG)** and **backmapping (CG → atomistic)**.

---

## ✨ Features

- **Command-Line and Library Interface**: Use `cgmap` as a simple command-line tool or import the `Mapper` class into your own Python scripts.  
- **Custom Mappings**: Define new coarse-graining schemes for any molecule using a simple YAML syntax.  
- **Trajectory Support**: Process entire MD trajectories (`.xtc`, `.trr`, etc.) in one go.  
- **Flexible I/O**: Supports any format readable by [MDAnalysis](https://www.mdanalysis.org/).  
- **Selection Language**: Use MDAnalysis selection strings (e.g., `"protein"`, `"resname POPC"`).  

---

## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/limresgrp/CGmap.git
cd CGmap
````

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Verify the installation:

```bash
cgmap test
```

⚡ **Optional dependency:** [aggforce](https://github.com/noegroup/aggforce)
Used to efficiently map atomistic forces to CG. Safe to ignore if you don’t need force mapping.

---

## ▶️ Usage

CGmap can be used in two ways:

1. **Command-Line Tool (CLI)**
2. **Python Library**

---

### 1. Command-Line Interface (CLI)

The `cgmap map` command is the main entry point.

**Syntax:**

```bash
cgmap map -m MAPPING -i INPUT [OPTIONS]
```

**Key Arguments:**

| Argument            | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `-m, --mapping`     | **(Required)** Mapping name (folder in `cgmap/data/` or custom path). |
| `-i, --input`       | **(Required)** Atomistic structure (`.pdb`, `.gro`).                  |
| `-o, --output`      | Output CG structure file. Defaults to `input_CG.gro`.                 |
| `-it, --inputtraj`  | One or more trajectory files (`.xtc`, `.trr`).                        |
| `-ot, --outputtraj` | Format for CG trajectory (default: `xtc`).                            |
| `-s, --selection`   | MDAnalysis selection string (default: `all`).                         |
| `-ts, --trajslice`  | Slice of trajectory (`start:stop:step`, e.g., `::10`).                |
| `--cg`              | Flag: input is already CG (for backmapping/validation/analysis).                  |

**Example:**

```bash
cgmap map \
    -m martini3 \
    -i system.pdb \
    -it trajectory.xtc \
    -o protein_cg.gro \
    -s "protein" \
    -ts "::10"
```

---

### 2. Python Library Usage

For more advanced workflows:

```python
from cgmap.mapping.mapper import Mapper

# 1. Define mapping arguments
args_dict = {
    'mapping': 'martini3',
    'input': 'path/to/system.pdb',
    'inputtraj': 'path/to/trajectory.xtc',
    'trajslice': '::10',      # Process every 10th frame
    'selection': 'protein',
    'isatomistic': True,      # Input is atomistic
}

# 2. Initialize the Mapper
mapper = Mapper(args_dict=args_dict)

# 3. Iterate and process each file/trajectory
for mapped_system, output_filename in mapper():
    if mapped_system:
        # Save coarse-grained outputs
        cg_structure, cg_traj = mapped_system.save(filename='output/system_cg.pdb')

        # Save original atomistic reference
        mapped_system.save_atomistic(filename='output/system_atomistic_ref.pdb')

        # Save everything in one dataset
        mapped_system.save_npz(
            filename='output/dataset.npz',
            to_pos_unit='Angstrom'
        )
```

---

## 🛠️ Mappings

### Creating a Custom Mapping

1. Create a new folder (e.g., `my_custom_mapping`).
2. Inside it, create one `.yaml` file for each residue/molecule. Example: `ALA.yaml`.

**Example (`ALA.yaml`):**

```yaml
molecule: ALA

atoms:
  H1:  BB  !     # Ignore this atom when building the CG bead
  H2:  BB  !
  H3:  BB  !
  N:   BB   P2AA # atom_name: bead_name P[level][anchorID][ID]. Used in HEroBM to backmap. Safely ignore providing P2AA if only mapping.
  H,HN:   BB     # Alternative atom names, depending on the force field
  CA:  BB   P1A
  HA:  BB   !
  CB:  SC1  P0A
  HB1: SC1  !
  HB2: SC1  !
  HB3: SC1  !
  C:   BB   P2AB
  O:   BB   P3BB
```

* **`molecule`**: Residue name CGmap will match.
* **`atoms`**: Maps atom names → bead names, plus optional metadata.

### Listing Available Mappings

Run the code snippet in the provided `tutorial.ipynb` to list available mappings and residues.

---

## 🤝 Contributing

Contributions are welcome! 🎉
Please open an issue or submit a pull request for improvements or new mappings.

---

## 📜 License

This project is licensed under the **MIT License**.