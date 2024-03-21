from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "cgmap/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="cgmap",
    version=version,
    description="CGmap is ...",
    download_url="https://github.com/limresgrp/CGmap",
    author="Daniele Angioletti",
    python_requires=">=3.7",
    packages=find_packages(include=["cgmap", "cgmap.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "cgmap = cgmap.scripts.map:main",
        ]
    },
    install_requires=[
        "numpy",
        "MDAnalysis",
        "pyyaml",
        "molmass",
        "contextlib2;python_version<'3.7'",  # backport of nullcontext
        'contextvars;python_version<"3.7"',  # backport of contextvars for savenload
        "typing_extensions;python_version<'3.8'",  # backport of Final
    ],
    zip_safe=True,
)
