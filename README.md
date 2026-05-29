# KAN-EAM/MEAM Interatomic Potential Framework


This repository contains a Python/PyTorch workflow for developing physically motivated machine-learning interatomic potentials by integrating Kolmogorov-Arnold Networks into EAM- and MEAM-like potential forms.

The project aims to combine the physical interpretability of classical interatomic potentials with the flexibility of machine learning, while maintaining compatibility with molecular dynamics workflows such as LAMMPS.

Example validation figures for an Ag test case are included to illustrate the training and validation workflow. The corresponding training data, trained checkpoints, and exported production potential are not included because they are part of ongoing unpublished work.

## Usage and License Notice

This repository is currently shared for academic demonstration and portfolio review purposes only.

The code and research materials are part of ongoing unpublished work. No permission is granted for reuse, redistribution, modification, or incorporation into other projects before publication of the corresponding paper.

After publication, reuse terms may be updated. Any future academic use should cite the corresponding paper and this repository.

See `NOTICE.md` for details.

## Motivation

Classical EAM and MEAM potentials are computationally efficient and physically interpretable, but their fixed analytical forms can limit flexibility for complex metallic systems. Fully neural-network-based potentials are flexible, but they may reduce interpretability and can require large training datasets.

This project explores a physically structured approach by replacing key EAM/MEAM functions, such as pair-interaction, electron-density, and embedding functions, with trainable KAN-based representations.

## Main Features

- VASP OUTCAR data processing for energy and force training data
- PyTorch-based model training
- Energy and force validation using parity plots
- Export of trained models to LAMMPS-compatible EAM/fs potential files
- Documentation for theory and workflow
- Template LAMMPS input files for exported-potential testing

## Repository Structure

```text
scripts/
    data_process.py      Convert VASP OUTCAR files into training data
    training.py          Train the KAN-EAM/MEAM model
    validate.py          Validate trained model and generate parity plots
    export_lammps.py     Export trained .pt model to LAMMPS EAM/fs format

docs/
    theory.md            Theory background of KAN-EAM/MEAM potentials
    workflow.md          Step-by-step workflow explanation

examples/
    demo_workflow/       Example workflow description and configuration template

lammps_examples/
    in.minimize_template Template LAMMPS minimization input

figures/
    Folder for validation figures and workflow diagrams
```

## Workflow

The typical workflow is:

1. Run `data_process.py` to convert VASP OUTCAR files into training data.
2. Run `training.py` to train the KAN-EAM/MEAM model.
3. Run `validate.py` to generate energy and force parity plots.
4. Run `export_lammps.py` to export the trained `.pt` model into a LAMMPS-compatible EAM/fs potential file.
5. Test the exported potential in LAMMPS.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate kan-eam-meam
```

Using pip:

```bash
pip install -r requirements.txt
```

## Example Usage

Data processing:

```bash
python scripts/data_process.py
```

Training:

```bash
python scripts/training.py
```

Validation:

```bash
python scripts/validate.py
```

Export to LAMMPS:

```bash
python scripts/export_lammps.py
```

## Documentation

For the physical model background, see:

```text
docs/theory.md
```

For the full workflow explanation, see:

```text
docs/workflow.md
```

## Notes

This repository is a research prototype. Full VASP datasets, trained checkpoints, exported production potentials, and unpublished collaborator data are not included.

## Research Context

This code was developed as part of research on machine-learning interatomic potentials for metallic materials, including Ag, NiTi, and high-entropy alloy systems.

## Author

Huang, Hung-Liang