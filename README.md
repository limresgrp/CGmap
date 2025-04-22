# CGmap

Map atomistic simulations to coarse-grained (CG) representations, with the flexibility to define your own custom CG mappings.

## Overview

This repository provides a Python script to map atomistic structures to coarse-grained (CG) representations. Users can define custom CG mappings, which may include additional metadata compatible with [HEroBM](https://github.com/limresgrp/HEroBM). This compatibility enables backmapping from CG to atomistic structures, facilitating seamless transitions between resolutions.

## Installation

To install CGmap, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/CGmap.git
    cd CGmap
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Verify the installation:
    ```bash
    cgmap test
    ```

## Usage

Here is a basic example of how to use CGmap:

1. Prepare your atomistic structure file (e.g., in PDB format).
2. Define your custom CG mapping in a configuration file.
3. Run the mapping script:
    ```bash
    python cgmap.py --input atomistic.pdb --mapping config.json --output cg.pdb
    ```

For more detailed usage instructions, refer to the documentation or examples provided in the repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.


## Acknowledgments

We would like to thank the following contributors for their valuable contributions:

- [@Dan Potemkin] for providing mapping configurations for RNA.cm.