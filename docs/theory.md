## EAM-like Energy Form

The embedded-atom method expresses the total energy of a metallic system as the sum of pair-interaction and embedding-energy contributions:

```math
E = \sum_{i \lt j} \phi_{ij}(r_{ij}) + \sum_i F_i(\rho_i)
```

where $\phi_{ij}(r_{ij})$ is the pair-interaction term between atoms $i$ and $j$, and $F_i(\rho_i)$ is the embedding energy of atom $i$ in the local electron-density environment.

The local electron density around atom $i$ is commonly written as:

```math
\rho_i = \sum_{j \ne i} \rho_j(r_{ij})
```

where $\rho_j(r_{ij})$ represents the electron-density contribution from neighboring atom $j$ at distance $r_{ij}$.


## KAN-based EAM/MEAM Representation

In this project, the functions \( \phi(r) \), \( \rho(r) \), and \( F(\rho) \) are represented using trainable Kolmogorov-Arnold Network functions.

The purpose is to improve the flexibility of classical EAM/MEAM-like potential forms while retaining a physically structured energy decomposition. Compared with fully black-box neural-network potentials, this approach keeps the model closer to classical interatomic-potential theory while allowing the functional forms to be learned from first-principles data.

## Validation

The trained potential should be evaluated using several criteria:

- energy RMSE,
- force RMSE,
- energy parity plots,
- force parity plots,
- smoothness of learned potential functions,
- consistency between PyTorch and LAMMPS energies,
- structural relaxation tests,
- short molecular dynamics stability tests.

## LAMMPS Export

After training, the learned functions are tabulated and exported into a LAMMPS-compatible EAM/fs potential file.

The exported potential should be tested by:

1. checking whether LAMMPS can read the potential file,
2. comparing LAMMPS energies with PyTorch-predicted energies,
3. minimizing simple crystal structures,
4. running short molecular dynamics simulations to check stability.