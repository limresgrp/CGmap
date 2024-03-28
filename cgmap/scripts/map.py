import sys
import argparse
import textwrap
import logging
from pathlib import Path
from typing import List

from cgmap.mapping.mapper import Mapper
from cgmap.scripts._logger import set_up_script_logger

import warnings
warnings.filterwarnings("ignore")


def main(args=None, running_as_script: bool = True):
    # in results dir, do: nequip-deploy build --train-dir . deployed.pth
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Map atomistic simulation to CG and save the CG structure in the desired format.
            """
        )
    )
    parser.add_argument(
        "-m",
        "--mapping",
        help="Name of the CG mapping.\nIt corresponds to the name of the chosen folder relative to the cgmap/data/ folder.\n"+
             "Optionally, the user can specify its own mapping folder by providing an absolute path.",
        type=Path,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input filename of atomistic structure to map.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html)",
        type=Path,
        default=None,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename of mapped CG structure.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html).\n" +
             "If not provided, the output filename will be the one of the input with the '_CG' suffix.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-it",
        "--inputtraj",
        nargs='+',
        help="List of trajectory files to load.\n",
        type=List[str],
        default=[],
    )
    parser.add_argument(
        "-ot",
        "--outputtraj",
        help="Format of the output trajectory (if multiple frames are mapped).\n" +
             "Defaults to xtc.",
        type=str,
        default='xtc',
    )
    parser.add_argument(
        "-ts",
        "--trajslice",
        help="Specify a slice of the total number of frames.\n" +
             "Only the sliced frames will be mapped.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms to map. Dafaults to 'all'.",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--log",
        help="log file.",
        type=Path,
        default=None,
    )
    
    # Something has to be provided
    # See https://stackoverflow.com/questions/22368458/how-to-make-argparse-logging.debug-usage-when-no-option-is-given-to-the-code
    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    # Parse the args
    args = parser.parse_args(args=args)

    # Setup the logger
    if running_as_script:
        set_up_script_logger(args.log)
    logger = logging.getLogger("cgmap-map")
    logger.setLevel(logging.INFO)

    # Do the mapping
    logger.info(f"Loading CG mappings from '{args.mapping}' folder")
    mapper = Mapper(vars(args), logger=logger)
    logger.info("Running CG mapping...")
    mapper.map()
    logger.info("Saving CG structure...")
    mapper.save()
    logger.info("Success!")

if __name__ == "__main__":
    main(running_as_script=True)