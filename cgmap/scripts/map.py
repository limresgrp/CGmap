import sys
import argparse
import textwrap
import logging
from pathlib import Path
from typing import Dict, List

from cgmap.mapping.mapper import Mapper
from cgmap.utils.logging import set_up_script_logger

import warnings
warnings.filterwarnings("ignore")


def aa2cg(args_dict: Dict):
    # Do the mapping
    print(f"Loading CG mappings from '{args_dict.get('mapping')}' folder")
    mapper = Mapper(args_dict)
    print("Running CG mapping...")
    cg_filenames, cg_filename_trajs = [], []
    for m, _ in mapper():
        print("Saving CG structure...")
        cg_filename, cg_filename_traj = m.save()
        cg_filenames.append(cg_filename)
        cg_filename_trajs.append(cg_filename_traj)
    print("Success!")
    return cg_filenames, cg_filename_trajs

def main(args=None, running_as_script: bool = True):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Map atomistic simulation to CG and save the CG structure in the desired format."""
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands: map | test")

    # Subparser for the 'map' command
    map_parser = subparsers.add_parser(
        "map",
        help="Perform the CG mapping."
    )
    map_parser.add_argument(
        "-m",
        "--mapping",
        help="Name of the CG mapping.\nIt corresponds to the name of the chosen folder relative to the cgmap/data/ folder.\n"+
             "Optionally, the user can specify its own mapping folder by providing an absolute path.",
        type=Path,
        required=True,
    )
    map_parser.add_argument(
        "-i",
        "--input",
        help="Input filename of atomistic structure to map.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html)",
        type=Path,
        required=True,
    )
    map_parser.add_argument(
        "-o",
        "--output",
        help="Output filename of mapped CG structure.\n" +
             "Supported file formats are those of MDAnalysis (see https://userguide.mdanalysis.org/stable/formats/index.html).\n" +
             "If not provided, the output filename will be the one of the input with the '_CG' suffix.",
        type=Path,
        default=None,
    )
    map_parser.add_argument(
        "-it",
        "--inputtraj",
        nargs='+',
        help="Trajectory file to load.\n",
        type=str,
        default=None,
    )
    map_parser.add_argument(
        "-ot",
        "--outputtraj",
        help="Format of the output trajectory (if multiple frames are mapped).\n" +
             "Defaults to xtc.",
        type=str,
        default='xtc',
    )
    map_parser.add_argument(
        "-ts",
        "--trajslice",
        help="Specify a slice of the total number of frames.\n" +
             "Only the sliced frames will be mapped.",
        type=str,
        default=None,
    )
    map_parser.add_argument(
        "-s",
        "--selection",
        help="Selection of atoms to map. Defaults to 'all'.",
        type=str,
        default="all",
    )
    map_parser.add_argument(
        "--log",
        help="Log file.",
        type=Path,
        default=None,
    )

    # Subparser for the 'test' command
    test_parser = subparsers.add_parser(
        "test",
        help="Test if all dependencies are correctly installed and exit."
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    if args.command == "test":
        # Handle the 'test' command
        print("Testing dependencies...")
        try:
            try:
                from aggforce import linearmap as lm
                from aggforce import agg as ag
                print("All dependencies are correctly installed!")
            except:
                print("aggforce is NOT installed!")
        except ImportError as e:
            print(f"Dependency error: {e}")
            sys.exit(1)
        sys.exit(0)

    elif args.command == "map":
        # Handle the 'map' command
        # Setup the logger
        if running_as_script:
            set_up_script_logger(args.log)
        logger = logging.getLogger("cgmap-map")
        logger.setLevel(logging.INFO)

        aa2cg(vars(args))


if __name__ == "__main__":
    main(running_as_script=True)