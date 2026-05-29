# LAMMPS Examples

This folder contains template LAMMPS input files for testing exported KAN-EAM/MEAM potentials.

The recommended validation steps are:

1. Check whether LAMMPS can read the exported potential.
2. Minimize a simple structure.
3. Compare the LAMMPS potential energy with the PyTorch model energy.
4. Run a short molecular dynamics simulation to test stability.

The included files are templates and may require modification depending on the exported potential name, element order, and structure file.