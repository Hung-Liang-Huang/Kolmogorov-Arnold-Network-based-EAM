# KAN-EAM Interatomic Potential Workflow

This repository contains a Python/PyTorch workflow for developing physically motivated machine-learning interatomic potentials by integrating Kolmogorov-Arnold Networks into EAM-like potential forms.

The current public version focuses on the KAN-EAM workflow, including VASP data processing, PyTorch training, model validation, and LAMMPS EAM/fs export. A KAN-MEAM extension is currently under development and will be added after successful testing and validation.

The project aims to combine the physical interpretability of classical interatomic potentials with the flexibility of machine learning, while maintaining compatibility with molecular dynamics workflows such as LAMMPS.

Example validation figures for an Ag test case are included to illustrate the training and validation workflow. The corresponding training data, trained checkpoints, and exported production potential are not included because they are part of ongoing unpublished work.

## Usage and License Notice

This repository is currently shared for academic demonstration and portfolio review purposes only.

The code and research materials are part of ongoing unpublished work. No permission is granted for reuse, redistribution, modification, or incorporation into other projects before publication of the corresponding paper.

After publication, reuse terms may be updated. Any future academic use should cite the corresponding paper and this repository.

See `NOTICE.md` for details.

## Motivation

Classical EAM potentials are computationally efficient and physically interpretable, but their fixed analytical forms can limit flexibility for complex metallic systems. Fully neural-network-based potentials are flexible, but they may reduce interpretability and can require large training datasets.

This project explores a physically structured approach by replacing key EAM functions, such as pair-interaction, electron-density, and embedding functions, with trainable KAN-based representations.

The longer-term goal is to extend this framework toward MEAM-like potential forms, where angular or environment-dependent contributions can be incorporated while retaining physical interpretability.

## Main Features

* VASP OUTCAR data processing for energy and force training data
* PyTorch-based KAN-EAM model training
* Energy and force validation using parity plots
* Export of trained models to LAMMPS-compatible EAM/fs potential files
* Documentation for theory and workflow
* Template LAMMPS input files for exported-potential testing
* Demonstration figures for an Ag test case

## Repository Structure

```text
scripts/
    data_process.py      Convert VASP OUTCAR files into training data
    training.py          Train the KAN-EAM model
    validate.py          Validate trained model and generate parity plots
    export_lammps.py     Export trained .pt model to LAMMPS EAM/fs format

docs/
    theory.md            Theory background of KAN-EAM potentials
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
2. Run `training.py` to train the KAN-EAM model.
3. Run `validate.py` to generate energy and force parity plots.
4. Run `export_lammps.py` to export the trained `.pt` model into a LAMMPS-compatible EAM/fs potential file.
5. Test the exported potential in LAMMPS.

## Recommended training database composition

To improve the stability and transferability of the KAN-EAM potential, the training database should include several physically important anchor and validation configurations. In the current training workflow, different database categories are assigned different energy and force weights through the `frame_anchor_weights()` function.

The recommended database categories are:

- `01_anchor_fcc_eos`: FCC equation-of-state structures. These configurations anchor the cohesive-energy curve and equilibrium lattice behavior.
- `02_anchor_fcc_small_displacement`: FCC small-displacement structures. These configurations strongly constrain harmonic force behavior around equilibrium.
- `03_anchor_fcc_elastic_strain`: FCC elastic-strain structures. These configurations improve the description of elastic response under homogeneous deformation.
- `04_fcc_finite_temperature_aimd`: finite-temperature FCC AIMD snapshots. These configurations improve robustness for thermally distorted local environments.
- `05_vacancy_and_point_defects`: vacancy and point-defect structures. These configurations improve defect energetics and local-environment transferability.
- `06_surfaces`: surface structures. These configurations improve the behavior of low-coordination atomic environments.

The current category-dependent energy and force weights are:

| Database category | Energy weight | Force weight |
|---|---:|---:|
| `01_anchor_fcc_eos` | 20.0 | 5.0 |
| `02_anchor_fcc_small_displacement` | 10.0 | 20.0 |
| `03_anchor_fcc_elastic_strain` | 15.0 | 10.0 |
| `04_fcc_finite_temperature_aimd` | 5.0 | 5.0 |
| `05_vacancy_and_point_defects` | 3.0 | 3.0 |
| `06_surfaces` | 2.0 | 3.0 |

The EOS and elastic-strain datasets are weighted more strongly for energy fitting, while the small-displacement dataset is weighted more strongly for force fitting. Defect, surface, and finite-temperature datasets are assigned moderate weights to improve transferability without overwhelming the bulk anchor data.

These datasets are recommended because relying only on near-equilibrium bulk structures may produce good parity performance on simple validation data but poor transferability to strained, defective, finite-temperature, or low-coordination configurations.

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

## Development Status

The current repository provides the tested KAN-EAM workflow. The KAN-MEAM implementation is under active development and will be uploaded after successful testing, validation, and documentation.

## Notes

This repository is a research prototype. Full VASP datasets, trained checkpoints, exported production potentials, and unpublished collaborator data are not included.

## Research Context

This code was developed as part of research on machine-learning interatomic potentials for metallic materials, including Ag, NiTi, and high-entropy alloy systems.

## Author

Huang, Hung-Liang
