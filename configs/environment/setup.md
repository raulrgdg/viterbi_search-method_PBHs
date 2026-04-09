Pipeline environment setup
==========================

Available export
----------------

The repository currently includes this Conda environment export:

- ../vit.yml

Environment name in the export:

- vit

Recommended reconstruction
--------------------------

From the repository root:

1. Create the Conda environment from the export:

   conda env create -f vit.yml

2. Activate it:

   conda activate vit

3. Verify key packages used by the pipeline:

   python -c "import pycbc, gwpy, lal, lalpulsar, soapcw"

Alternative update path
-----------------------

If the environment already exists and you want to refresh it from the export:

   conda env update -n vit -f vit.yml --prune

What this environment contains
------------------------------

The export includes, among others:

- Python 3.10
- pycbc
- gwpy
- lalsuite
- soap / soapcw
- htcondor
- numpy, scipy, pandas, matplotlib
- torch

Notes
-----

- The export was generated from a Linux Conda environment and includes a `prefix` field pointing to the original machine. Conda will ignore that old path when creating a new environment elsewhere.
- The environment mixes Conda packages with a `pip:` section inside `vit.yml`; use the YAML file directly instead of manually recreating packages one by one.
- If `conda env create -f vit.yml` fails on another platform, start from Python 3.10 and then install the scientific stack incrementally, prioritizing `lalsuite`, `pycbc`, `gwpy`, and `soapcw`.

Useful commands
---------------

Inspect available environments:

   conda env list

Export the active environment again later:

   conda env export -n vit --no-builds > vit.yml
