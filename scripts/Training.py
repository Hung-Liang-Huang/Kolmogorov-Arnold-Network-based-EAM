
"""
training.py

Train a KAN-based EAM/MEAM interatomic potential using processed
DFT energy and force data.

Main tasks:
    1. Load processed training data.
    2. Build the KAN-EAM/MEAM model.
    3. Train using energy and force losses.
    4. Save trained model checkpoints.

Author:
    Huang, Hung-Liang
"""


from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# KAN-EAM Training for Ag Stage 1
# ============================================================
#
# Training database:
#   training_data_Ag_stage1/
#       01_anchor_fcc_eos/
#       02_anchor_fcc_small_displacement/
#       03_anchor_fcc_elastic_strain/
#
# Main purpose:
#   Learn a stable fcc Ag energy landscape before adding defects,
#   surfaces, stacking faults, liquid, or amorphous configurations.
#
# Model:
#   E = sum_i F(rhobar_i) + sum_{i<j} phi(r_ij)
#   rhobar_i = sum_j rho(r_ij)
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(r"./training_data_Ag_stage1")
OUT_ROOT = Path(r"./training_out_kan_eam_Ag_stage1_01_02_03")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

CKPT_DIR = OUT_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_DIR = OUT_ROOT / "exported_functions"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL SETTINGS
# ============================================================

SEED = 1234

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

TRAIN_FRAC = 0.90

EPOCHS = 600
LR = 5.0e-4
WEIGHT_DECAY = 0.0

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
CLIP_GRAD_NORM = 1.0

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 100
EARLY_STOPPING_MIN_DELTA = 1.0e-4
MIN_EPOCHS_BEFORE_STOP = 200

USE_HUBER_FORCES = False
FORCE_HUBER_BETA = 0.05


# ============================================================
# Ag KAN-EAM MODEL SETTINGS
# ============================================================

# Ag nearest-neighbor distance is around 2.9 Å.
# 6.2 Å includes several neighbor shells.
RCUT = 6.2
RMIN = 1.8

RHOMIN = 0.0
RHOMAX = 30.0

N_CTRL_R = 80
N_CTRL_F = 14
N_CTRL_RHO = 80

USE_ZBL = True
ZBL_R_SWITCH = 2.2

PRINT_RHOBAR_EVERY = 5
RHO_WARN_FRAC = 0.95


# ============================================================
# LOSS WEIGHTS
# ============================================================

W_E = 1.0

W_F_STAGE1 = 1.0
W_F_STAGE2 = 2.0
W_F_STAGE3 = 2.0

W_REG_STAGE1 = 0.40
W_REG_STAGE2 = 0.80
W_REG_STAGE3 = 0.80

FREEZE_F_EPOCHS = 30

# Balanced score reference values
E_REF = 1.0e-3
F_REF = 1.0e-1
ALPHA_BAL = 1.0


# ============================================================
# REGULARIZATION
# ============================================================

# Embedding F
LAMBDA_SMOOTH_F = 2.0e-3
LAMBDA_F_GAUGE = 5.0
LAMBDA_F_SLOPE0 = 1.0e-2

LAMBDA_F_WINDOW_POSSLOPE = 8.0e-2
LAMBDA_F_WINDOW_CURV = 2.0e-3
LAMBDA_F_WINDOW_SIGNTV = 2.0e-2

F_SIGN_BETA = 6.0
ALLOWED_SIGN_TV = 2.2

LAMBDA_F_OUTSIDE_SMOOTH = 3.0e-4
ACTIVE_WINDOW_PAD = 1.5
F_GRID = 1024

# Density rho
LAMBDA_SMOOTH_RHO = 1.0e-3
LAMBDA_MONO_RHO = 1.0e-1
LAMBDA_RHO_RC_VAL = 8.0
LAMBDA_RHO_RC_SLOPE = 4.0
LAMBDA_RHO_TAIL_FLAT = 1.0e-1
MONO_GRID = 512

# Pair phi
LAMBDA_SMOOTH_PHI = 5.0e-4
LAMBDA_PHI_RC_VAL = 8.0
LAMBDA_PHI_RC_SLOPE = 4.0
LAMBDA_PHI_TAIL_FLAT = 1.0e-1


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smooth_cutoff(r: torch.Tensor, rc: float) -> torch.Tensor:
    x = r / rc
    return torch.where(
        x < 1.0,
        1.0 - 3.0 * x * x + 2.0 * x * x * x,
        torch.zeros_like(x),
    )


def smoothstep01(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def zbl_phi(r: torch.Tensor, Z1: float, Z2: float) -> torch.Tensor:
    """
    Universal ZBL short-range repulsion.
    Energy unit: eV
    Distance unit: Angstrom
    """
    k = 14.399645
    a0 = 0.529177

    a = 0.88534 * a0 / (Z1**0.23 + Z2**0.23)
    x = r / a

    c = torch.tensor(
        [0.1818, 0.5099, 0.2802, 0.02817],
        device=r.device,
        dtype=r.dtype,
    )
    d = torch.tensor(
        [3.2, 0.9423, 0.4029, 0.2016],
        device=r.device,
        dtype=r.dtype,
    )

    Phi = (
        c[0] * torch.exp(-d[0] * x)
        + c[1] * torch.exp(-d[1] * x)
        + c[2] * torch.exp(-d[2] * x)
        + c[3] * torch.exp(-d[3] * x)
    )

    r_safe = torch.clamp(r, min=1.0e-3)
    return k * (Z1 * Z2) * Phi / r_safe


# ============================================================
# ANCHOR WEIGHTS
# ============================================================

def frame_anchor_weights(path: str) -> Tuple[float, float]:
    """
    Return energy and force weights according to the source folder.

    These weights are important for Ag stage-1 training.

    01 fcc EOS:
        Strong energy weight because EOS determines equilibrium volume
        and bulk modulus.

    02 small displacement:
        Strong force weight because this determines whether fcc atoms
        are restored to correct lattice sites.

    03 elastic strain:
        Strong energy and force weights because this determines elastic
        and box stability.
    """
    p = path.replace("\\", "/").lower()

    if "01_anchor_fcc_eos" in p:
        return 20.0, 5.0

    if "02_anchor_fcc_small_displacement" in p:
        return 10.0, 20.0

    if "03_anchor_fcc_elastic_strain" in p:
        return 15.0, 10.0

    return 1.0, 1.0


# ============================================================
# UNIFORM CUBIC B-SPLINE
# ============================================================

class UniformCubicBSpline1D(nn.Module):
    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_ctrl: int,
        init_y: Optional[np.ndarray] = None,
    ):
        super().__init__()

        assert n_ctrl >= 4

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.n_ctrl = int(n_ctrl)
        self.h = (self.x_max - self.x_min) / (self.n_ctrl - 3)

        if init_y is None:
            c0 = 0.01 * torch.randn(self.n_ctrl, dtype=DTYPE)
        else:
            c0 = torch.tensor(init_y, dtype=DTYPE)

        self.ctrl = nn.Parameter(c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.x_min, self.x_max)

        u = (x - self.x_min) / self.h
        idx = torch.floor(u).to(torch.long)
        t = u - idx.to(u.dtype)

        idx = torch.clamp(idx, 0, self.n_ctrl - 4)

        t2 = t * t
        t3 = t2 * t

        B0 = (1.0 - t) ** 3 / 6.0
        B1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        B2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        B3 = t3 / 6.0

        c = self.ctrl

        return (
            c[idx + 0] * B0
            + c[idx + 1] * B1
            + c[idx + 2] * B2
            + c[idx + 3] * B3
        )

    def smoothness_penalty(self) -> torch.Tensor:
        c = self.ctrl
        d2 = c[:-2] - 2.0 * c[1:-1] + c[2:]
        return torch.mean(d2 * d2)

    def monotone_decreasing_penalty(self, n_grid: int = 512) -> torch.Tensor:
        x = torch.linspace(
            self.x_min,
            self.x_max,
            n_grid,
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y = self.forward(x)
        dx = (self.x_max - self.x_min) / (n_grid - 1)
        dydx = (y[1:] - y[:-1]) / dx

        return torch.mean(torch.relu(dydx) ** 2)

    def positive_slope_penalty_window(
        self,
        x_lo: float,
        x_hi: float,
        n_grid: int = 256,
    ) -> torch.Tensor:
        x_lo = max(self.x_min, float(x_lo))
        x_hi = min(self.x_max, float(x_hi))

        if x_hi <= x_lo + 1.0e-12:
            return torch.zeros((), device=self.ctrl.device, dtype=self.ctrl.dtype)

        x = torch.linspace(
            x_lo,
            x_hi,
            n_grid,
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y = self.forward(x)
        dx = (x_hi - x_lo) / (n_grid - 1)
        dydx = (y[1:] - y[:-1]) / dx

        return torch.mean(torch.relu(dydx) ** 2)

    def curvature_penalty_window(
        self,
        x_lo: float,
        x_hi: float,
        n_grid: int = 256,
    ) -> torch.Tensor:
        x_lo = max(self.x_min, float(x_lo))
        x_hi = min(self.x_max, float(x_hi))

        if x_hi <= x_lo + 1.0e-12:
            return torch.zeros((), device=self.ctrl.device, dtype=self.ctrl.dtype)

        x = torch.linspace(
            x_lo,
            x_hi,
            n_grid,
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y = self.forward(x)
        d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]

        return torch.mean(d2 * d2)

    def sign_change_penalty_window(
        self,
        x_lo: float,
        x_hi: float,
        n_grid: int = 256,
        beta: float = 6.0,
        allowed_tv: float = 2.2,
    ) -> torch.Tensor:
        x_lo = max(self.x_min, float(x_lo))
        x_hi = min(self.x_max, float(x_hi))

        if x_hi <= x_lo + 1.0e-12:
            return torch.zeros((), device=self.ctrl.device, dtype=self.ctrl.dtype)

        x = torch.linspace(
            x_lo,
            x_hi,
            n_grid,
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y = self.forward(x)
        dx = (x_hi - x_lo) / (n_grid - 1)
        dydx = (y[1:] - y[:-1]) / dx

        soft_sign = torch.tanh(beta * dydx)
        tv = torch.sum(torch.abs(soft_sign[1:] - soft_sign[:-1]))

        return torch.relu(tv - allowed_tv) ** 2

    def right_value_and_slope_penalty(
        self,
        target_value: float = 0.0,
        target_slope: float = 0.0,
        dx_frac: float = 1.0e-3,
    ):
        dx = max((self.x_max - self.x_min) * dx_frac, 1.0e-6)

        x1 = torch.tensor(
            [self.x_max - dx],
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )
        x2 = torch.tensor(
            [self.x_max],
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y1 = self.forward(x1)[0]
        y2 = self.forward(x2)[0]

        slope = (y2 - y1) / dx

        p_val = (y2 - target_value) ** 2
        p_slope = (slope - target_slope) ** 2

        return p_val, p_slope

    def tail_flatness_penalty(
        self,
        frac_from_right: float = 0.10,
        n_grid: int = 128,
    ):
        x_lo = self.x_max - frac_from_right * (self.x_max - self.x_min)

        x = torch.linspace(
            x_lo,
            self.x_max,
            n_grid,
            device=self.ctrl.device,
            dtype=self.ctrl.dtype,
        )

        y = self.forward(x)
        d2 = y[:-2] - 2.0 * y[1:-1] + y[2:]

        return torch.mean(d2 * d2)

    def export_table(self, n_grid: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.linspace(
                self.x_min,
                self.x_max,
                n_grid,
                device=self.ctrl.device,
                dtype=self.ctrl.dtype,
            )
            y = self.forward(x)

        return x.cpu().numpy(), y.cpu().numpy()


# ============================================================
# SPECIES AND DATASET
# ============================================================

@dataclass
class SpeciesInfo:
    z_to_type: Dict[int, int]
    type_to_z: Dict[int, int]
    n_types: int


def make_species_info(all_atomic_numbers: List[int]) -> SpeciesInfo:
    uniq = sorted(set(int(z) for z in all_atomic_numbers))

    z_to_type = {z: i for i, z in enumerate(uniq)}
    type_to_z = {i: z for z, i in z_to_type.items()}

    return SpeciesInfo(
        z_to_type=z_to_type,
        type_to_z=type_to_z,
        n_types=len(uniq),
    )


def find_all_npz_files(data_root: Path) -> List[Path]:
    files = sorted([p for p in data_root.rglob("*.npz") if p.is_file()])

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found under: {data_root}")

    return files


def scan_species_from_npz_files(data_root: Path) -> List[int]:
    zs = []

    for npz_path in find_all_npz_files(data_root):
        d = np.load(npz_path, allow_pickle=True)
        zs.extend([int(z) for z in d["numbers"].astype(np.int32).tolist()])

    return zs


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: Path, species_info: SpeciesInfo):
        self.files = find_all_npz_files(data_root)
        self.sp = species_info

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        npz_path = self.files[idx]
        d = np.load(npz_path, allow_pickle=True)

        cell = d["cell"].astype(np.float64)
        pbc = d["pbc"].astype(np.bool_)
        pos = d["positions"].astype(np.float64)
        numbers = d["numbers"].astype(np.int32)

        energy = float(d["energy"])
        forces = d["forces"].astype(np.float64) if "forces" in d.files else None

        types = np.array(
            [self.sp.z_to_type[int(z)] for z in numbers],
            dtype=np.int64,
        )

        return {
            "cell": cell,
            "pbc": pbc,
            "pos": pos,
            "types": types,
            "energy": energy,
            "forces": forces,
            "natoms": pos.shape[0],
            "path": str(npz_path),
        }


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def estimate_scales(
    ds: torch.utils.data.Dataset,
    n_sample: int = 200,
) -> Tuple[float, float]:
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(n_sample, len(ds))]

    e_list = []
    f_list = []

    for i in idxs:
        b = ds[i]
        nat = b["natoms"]

        e_list.append(b["energy"] / nat)

        if b["forces"] is not None:
            f_list.append(np.ravel(b["forces"]))

    e_arr = np.array(e_list, dtype=np.float64)

    if len(f_list) > 0:
        f_arr = np.concatenate(f_list).astype(np.float64)
    else:
        f_arr = np.array([1.0], dtype=np.float64)

    e_scale = float(np.sqrt(np.mean((e_arr - np.mean(e_arr)) ** 2)) + 1.0e-12)
    f_scale = float(np.sqrt(np.mean(f_arr ** 2)) + 1.0e-12)

    return e_scale, f_scale


# ============================================================
# Ag-SPECIFIC INITIALIZATION
# ============================================================

def init_rho_ctrl(n_ctrl: int, x_min: float, x_max: float) -> np.ndarray:
    """
    Smooth decaying density for Ag.
    """
    xs = np.linspace(x_min, x_max, n_ctrl)

    beta = 1.2
    y = np.exp(-beta * (xs - 2.5))
    y = y / (y.max() + 1.0e-12)
    y = 0.6 * y

    return y


def init_phi_ctrl(n_ctrl: int, x_min: float, x_max: float) -> np.ndarray:
    """
    Ag-like Morse-style pair initialization.
    This is only an initial guess.
    """
    xs = np.linspace(x_min, x_max, n_ctrl)

    r0 = 2.90
    D = 0.12
    a = 1.6

    y = D * (
        np.exp(-2.0 * a * (xs - r0))
        - 2.0 * np.exp(-a * (xs - r0))
    )

    tail = 1.0 - ((xs - x_min) / (x_max - x_min)) ** 2
    tail = np.clip(tail, 0.0, 1.0)

    y = y * tail

    return y


def init_F_ctrl(n_ctrl: int, x_min: float, x_max: float) -> np.ndarray:
    """
    Standard EAM-like attractive embedding initialization.
    """
    xs = np.linspace(x_min, x_max, n_ctrl)

    y = -0.35 * np.sqrt(np.clip(xs, 0.0, None) + 1.0e-8)

    # Gauge: F(0) = 0
    y = y - y[0]

    return y


# ============================================================
# KAN-EAM MODEL
# ============================================================

class KANEAM_Cubic(nn.Module):
    def __init__(self, sp: SpeciesInfo):
        super().__init__()

        self.sp = sp

        self.rmin = float(RMIN)
        self.rcut = float(RCUT)
        self.rhomax = float(RHOMAX)

        self.register_buffer("active_rho_min", torch.tensor(8.0, dtype=DTYPE))
        self.register_buffer("active_rho_max", torch.tensor(12.0, dtype=DTYPE))

        T = sp.n_types

        F_init = init_F_ctrl(N_CTRL_F, RHOMIN, RHOMAX)
        rho_init = init_rho_ctrl(N_CTRL_R, RMIN, RCUT)
        phi_init = init_phi_ctrl(N_CTRL_R, RMIN, RCUT)

        self.F_embed = nn.ModuleList(
            [
                UniformCubicBSpline1D(
                    RHOMIN,
                    RHOMAX,
                    N_CTRL_F,
                    init_y=F_init.copy(),
                )
                for _ in range(T)
            ]
        )

        self.rho = nn.ModuleList(
            [
                UniformCubicBSpline1D(
                    RMIN,
                    RCUT,
                    N_CTRL_R,
                    init_y=rho_init.copy(),
                )
                for _ in range(T * T)
            ]
        )

        self.phi = nn.ModuleList(
            [
                UniformCubicBSpline1D(
                    RMIN,
                    RCUT,
                    N_CTRL_R,
                    init_y=phi_init.copy(),
                )
                for _ in range(T * (T + 1) // 2)
            ]
        )

    def idx_rho(self, a: int, b: int) -> int:
        return a * self.sp.n_types + b

    def idx_phi(self, a: int, b: int) -> int:
        if a > b:
            a, b = b, a

        T = self.sp.n_types

        return a * T - (a * (a - 1)) // 2 + (b - a)

    def eval_rho(self, a: int, b: int, r: torch.Tensor) -> torch.Tensor:
        s = self.rho[self.idx_rho(a, b)]

        base = F.softplus(s.forward(r))

        return base * smooth_cutoff(r, self.rcut)

    def eval_phi(self, a: int, b: int, r: torch.Tensor) -> torch.Tensor:
        s = self.phi[self.idx_phi(a, b)]

        base = s.forward(r) * smooth_cutoff(r, self.rcut)

        if not USE_ZBL:
            return base

        Z1 = float(self.sp.type_to_z[a])
        Z2 = float(self.sp.type_to_z[b])

        z = zbl_phi(r, Z1, Z2)

        rs = float(ZBL_R_SWITCH)
        r0 = 0.8 * rs
        r1 = 1.2 * rs

        t = torch.clamp((r - r0) / (r1 - r0 + 1.0e-12), 0.0, 1.0)
        t = smoothstep01(t)

        return (1.0 - t) * z + t * base

    def forward_energy_forces(
        self,
        cell,
        pbc,
        pos,
        types,
        return_rhobar=False,
        return_parts=False,
    ):
        N = pos.shape[0]

        inv_cell = torch.linalg.inv(cell)
        frac = pos @ inv_cell

        df = frac[:, None, :] - frac[None, :, :]

        if bool(pbc[0]):
            df[..., 0] = df[..., 0] - torch.round(df[..., 0])
        if bool(pbc[1]):
            df[..., 1] = df[..., 1] - torch.round(df[..., 1])
        if bool(pbc[2]):
            df[..., 2] = df[..., 2] - torch.round(df[..., 2])

        dr = df @ cell
        rij = torch.linalg.norm(dr + 1.0e-18, dim=-1)

        eye = torch.eye(N, device=pos.device, dtype=torch.bool)

        m = (~eye) & (rij < self.rcut) & (rij > self.rmin)

        rhobar = torch.zeros((N,), device=pos.device, dtype=pos.dtype)

        T = self.sp.n_types

        # ------------------------------------------------------------
        # Density term
        # rho_{site_type, neighbor_type}
        # ------------------------------------------------------------
        for a in range(T):
            ia = types == a

            if not torch.any(ia):
                continue

            for b in range(T):
                ib = types == b

                if not torch.any(ib):
                    continue

                mask_ab = ia[:, None] & ib[None, :] & m

                if not torch.any(mask_ab):
                    continue

                r_ab = rij[mask_ab]
                rho_ab = self.eval_rho(a, b, r_ab)

                idx_i = torch.nonzero(mask_ab, as_tuple=False)[:, 0]
                rhobar.index_add_(0, idx_i, rho_ab)

        rhobar = torch.clamp(rhobar, 0.0, self.rhomax)

        with torch.no_grad():
            self.active_rho_min.copy_(rhobar.min().detach())
            self.active_rho_max.copy_(rhobar.max().detach())

        # ------------------------------------------------------------
        # Embedding energy
        # ------------------------------------------------------------
        E_embed_atoms = torch.zeros((N,), device=pos.device, dtype=pos.dtype)

        for a in range(T):
            ia = types == a

            if torch.any(ia):
                E_embed_atoms[ia] = self.F_embed[a].forward(rhobar[ia])

        E_embed_total = torch.sum(E_embed_atoms)

        # ------------------------------------------------------------
        # Pair energy
        # Count each physical pair once with upper triangular mask.
        # For multi-element systems, cross-pair orientations are both included.
        # For Ag single element this still works.
        # ------------------------------------------------------------
        iu = torch.triu(
            torch.ones((N, N), device=pos.device, dtype=torch.bool),
            diagonal=1,
        )

        m_u = m & iu

        type_i = types[:, None]
        type_j = types[None, :]

        E_pair = torch.zeros((), device=pos.device, dtype=pos.dtype)

        for a in range(T):
            for b in range(a, T):
                if a == b:
                    mask_ab = (type_i == a) & (type_j == a) & m_u
                else:
                    mask_ab = (
                        (
                            ((type_i == a) & (type_j == b))
                            | ((type_i == b) & (type_j == a))
                        )
                        & m_u
                    )

                if not torch.any(mask_ab):
                    continue

                r_ab = rij[mask_ab]
                phi_ab = self.eval_phi(a, b, r_ab)

                E_pair = E_pair + torch.sum(phi_ab)

        E_total = E_embed_total + E_pair

        forces = -torch.autograd.grad(
            E_total,
            pos,
            create_graph=True,
            retain_graph=True,
        )[0]

        if return_parts:
            if return_rhobar:
                return (
                    E_total,
                    forces,
                    E_embed_total.detach(),
                    E_pair.detach(),
                    rhobar.detach(),
                )
            return E_total, forces, E_embed_total.detach(), E_pair.detach()

        if return_rhobar:
            return E_total, forces, rhobar.detach()

        return E_total, forces

    def regularization(self) -> torch.Tensor:
        reg = torch.zeros(
            (),
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )

        rho_lo = max(RHOMIN, float(self.active_rho_min.item()) - ACTIVE_WINDOW_PAD)
        rho_hi = min(RHOMAX, float(self.active_rho_max.item()) + ACTIVE_WINDOW_PAD)

        # ------------------------------------------------------------
        # Embedding F
        # ------------------------------------------------------------
        for f in self.F_embed:
            reg = reg + LAMBDA_F_OUTSIDE_SMOOTH * f.smoothness_penalty()

            x0 = torch.tensor(
                [RHOMIN],
                device=f.ctrl.device,
                dtype=f.ctrl.dtype,
            )
            f0 = f.forward(x0)[0]

            reg = reg + LAMBDA_F_GAUGE * (f0 ** 2)

            xg = torch.tensor(
                [RHOMIN, RHOMIN + 1.0e-2],
                device=f.ctrl.device,
                dtype=f.ctrl.dtype,
            )

            yg = f.forward(xg)
            df0 = (yg[1] - yg[0]) / 1.0e-2

            reg = reg + LAMBDA_F_SLOPE0 * (df0 ** 2)

            reg = reg + LAMBDA_F_WINDOW_POSSLOPE * f.positive_slope_penalty_window(
                rho_lo,
                rho_hi,
                n_grid=512,
            )

            reg = reg + LAMBDA_F_WINDOW_CURV * f.curvature_penalty_window(
                rho_lo,
                rho_hi,
                n_grid=512,
            )

            reg = reg + LAMBDA_F_WINDOW_SIGNTV * f.sign_change_penalty_window(
                rho_lo,
                rho_hi,
                n_grid=512,
                beta=F_SIGN_BETA,
                allowed_tv=ALLOWED_SIGN_TV,
            )

        # ------------------------------------------------------------
        # Density rho
        # ------------------------------------------------------------
        for s in self.rho:
            reg = reg + LAMBDA_SMOOTH_RHO * s.smoothness_penalty()
            reg = reg + LAMBDA_MONO_RHO * s.monotone_decreasing_penalty(
                n_grid=MONO_GRID
            )

            p_val, p_slope = s.right_value_and_slope_penalty(0.0, 0.0)

            reg = reg + LAMBDA_RHO_RC_VAL * p_val
            reg = reg + LAMBDA_RHO_RC_SLOPE * p_slope
            reg = reg + LAMBDA_RHO_TAIL_FLAT * s.tail_flatness_penalty(
                frac_from_right=0.12,
                n_grid=128,
            )

        # ------------------------------------------------------------
        # Pair phi
        # ------------------------------------------------------------
        for s in self.phi:
            reg = reg + LAMBDA_SMOOTH_PHI * s.smoothness_penalty()

            p_val, p_slope = s.right_value_and_slope_penalty(0.0, 0.0)

            reg = reg + LAMBDA_PHI_RC_VAL * p_val
            reg = reg + LAMBDA_PHI_RC_SLOPE * p_slope
            reg = reg + LAMBDA_PHI_TAIL_FLAT * s.tail_flatness_penalty(
                frac_from_right=0.12,
                n_grid=128,
            )

        return reg

    def export_all(
        self,
        out_npz: Path,
        n_grid_r: int = 4096,
        n_grid_rho: int = 4096,
    ):
        T = self.sp.n_types

        r_grid = np.linspace(self.rmin, self.rcut, n_grid_r)
        rho_grid = np.linspace(RHOMIN, RHOMAX, n_grid_rho)

        data = {
            "r_grid": r_grid,
            "rho_grid": rho_grid,
            "type_to_z": np.array(
                [self.sp.type_to_z[i] for i in range(T)],
                dtype=np.int32,
            ),
            "rmin": np.array(self.rmin, dtype=np.float64),
            "rcut": np.array(self.rcut, dtype=np.float64),
            "rhomin": np.array(RHOMIN, dtype=np.float64),
            "rhomax": np.array(RHOMAX, dtype=np.float64),
        }

        for a in range(T):
            x, y = self.F_embed[a].export_table(n_grid_rho)

            data[f"F_type{a}_x"] = x
            data[f"F_type{a}_y"] = y
            data[f"F_type{a}_ctrl"] = self.F_embed[a].ctrl.detach().cpu().numpy()

        rr = torch.tensor(
            r_grid,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )

        with torch.no_grad():
            for a in range(T):
                for b in range(T):
                    s = self.rho[self.idx_rho(a, b)]

                    y = F.softplus(s.forward(rr)) * smooth_cutoff(rr, self.rcut)

                    data[f"rho_{b}_to_{a}_y"] = y.cpu().numpy()
                    data[f"rho_{b}_to_{a}_ctrl"] = s.ctrl.detach().cpu().numpy()

            for a in range(T):
                for b in range(a, T):
                    y = self.eval_phi(a, b, rr)

                    data[f"phi_{a}_{b}_y"] = y.cpu().numpy()
                    data[f"phi_{a}_{b}_ctrl"] = (
                        self.phi[self.idx_phi(a, b)]
                        .ctrl.detach()
                        .cpu()
                        .numpy()
                    )

        np.savez_compressed(out_npz, **data)


# ============================================================
# PLOTTING
# ============================================================

def plot_export_npz(
    export_npz: Path,
    out_dir: Path,
    tag: str,
    dump_all: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(export_npz, allow_pickle=True)

    r_grid = d["r_grid"]
    rho_grid = d["rho_grid"]
    type_to_z = d["type_to_z"].astype(int)
    T = len(type_to_z)

    plt.figure()
    for a in range(T):
        plt.plot(
            rho_grid,
            d[f"F_type{a}_y"],
            label=f"type {a} Z={type_to_z[a]}",
        )
    plt.xlabel(r"$\bar{\rho}$")
    plt.ylabel("F (eV)")
    plt.title(f"Embedding F ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"SUMMARY_F_{tag}.png", dpi=200)
    plt.close()

    plt.figure()
    for a in range(T):
        plt.plot(
            r_grid,
            d[f"rho_{a}_to_{a}_y"],
            label=f"{a}->{a}",
        )
    plt.xlabel("r (Angstrom)")
    plt.ylabel("rho(r)")
    plt.title(f"rho self-pair ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"SUMMARY_rho_self_{tag}.png", dpi=200)
    plt.close()

    plt.figure()
    for a in range(T):
        plt.plot(
            r_grid,
            d[f"phi_{a}_{a}_y"],
            label=f"({a},{a})",
        )
    plt.xlabel("r (Angstrom)")
    plt.ylabel("phi(r) (eV)")
    plt.title(f"phi self-pair ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"SUMMARY_phi_self_{tag}.png", dpi=200)
    plt.close()

    if not dump_all:
        return

    full_dir = out_dir / f"all_curves_{tag}"
    full_dir.mkdir(parents=True, exist_ok=True)

    for a in range(T):
        plt.figure()
        plt.plot(rho_grid, d[f"F_type{a}_y"])
        plt.xlabel(r"$\bar{\rho}$")
        plt.ylabel("F (eV)")
        plt.title(f"F type {a} Z={type_to_z[a]} [{tag}]")
        plt.tight_layout()
        plt.savefig(full_dir / f"F_type{a}_Z{type_to_z[a]}.png", dpi=200)
        plt.close()

    for a in range(T):
        for b in range(T):
            plt.figure()
            plt.plot(r_grid, d[f"rho_{b}_to_{a}_y"])
            plt.xlabel("r (Angstrom)")
            plt.ylabel("rho(r)")
            plt.title(f"rho {b}->{a} [{tag}]")
            plt.tight_layout()
            plt.savefig(full_dir / f"rho_b{b}_to_a{a}.png", dpi=200)
            plt.close()

    for a in range(T):
        for b in range(a, T):
            plt.figure()
            plt.plot(r_grid, d[f"phi_{a}_{b}_y"])
            plt.xlabel("r (Angstrom)")
            plt.ylabel("phi(r) (eV)")
            plt.title(f"phi {a}-{b} [{tag}]")
            plt.tight_layout()
            plt.savefig(full_dir / f"phi_a{a}_b{b}.png", dpi=200)
            plt.close()


def plot_metric_history(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray(history["epoch"], dtype=np.int32)

    if len(epochs) == 0:
        return

    items = [
        (
            "loss_history.png",
            "Loss",
            [
                (history["train_loss"], "train loss"),
                (history["val_loss"], "val loss"),
            ],
        ),
        (
            "E_RMSE_history.png",
            "E_RMSE (eV/atom)",
            [
                (history["train_e_rmse"], "train E_RMSE"),
                (history["val_e_rmse"], "val E_RMSE"),
            ],
        ),
        (
            "F_RMSE_history.png",
            "F_RMSE (eV/Angstrom)",
            [
                (history["train_f_rmse"], "train F_RMSE"),
                (history["val_f_rmse"], "val F_RMSE"),
            ],
        ),
        (
            "BALANCED_history.png",
            "Balanced score",
            [
                (history["balanced_score"], "val balanced score"),
            ],
        ),
        (
            "lr_history.png",
            "Learning rate",
            [
                (history["lr"], "learning rate"),
            ],
        ),
    ]

    for fname, ylabel, series in items:
        plt.figure()

        for arr, label in series:
            plt.plot(epochs, arr, label=label)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(fname.replace("_", " ").replace(".png", ""))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()


def save_history_npz(history: dict, out_path: Path):
    np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in history.items()})


# ============================================================
# RHOBAR ESTIMATE
# ============================================================

def estimate_rhobar_range(ds, sp, n_sample=50):
    tmp_model = KANEAM_Cubic(sp).to(DEVICE).to(DTYPE)

    mins = []
    means = []
    maxs = []

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(n_sample, len(ds))]

    for i in idxs:
        b = ds[i]

        cell = torch.tensor(b["cell"], device=DEVICE, dtype=DTYPE)
        pbc = torch.tensor(b["pbc"], device=DEVICE)
        pos = torch.tensor(
            b["pos"],
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        types = torch.tensor(b["types"], device=DEVICE, dtype=torch.long)

        _, _, rh = tmp_model.forward_energy_forces(
            cell,
            pbc,
            pos,
            types,
            return_rhobar=True,
        )

        mins.append(float(rh.min().cpu()))
        means.append(float(rh.mean().cpu()))
        maxs.append(float(rh.max().cpu()))

    return min(mins), float(np.mean(means)), max(maxs)


# ============================================================
# RUNNING METRICS
# ============================================================

@dataclass
class Running:
    nE: int = 0
    seE: float = 0.0
    nF: int = 0
    seF: float = 0.0

    def add_energy(self, err: torch.Tensor):
        self.nE += 1
        self.seE += float((err * err).detach().cpu())

    def add_forces(self, ferr: torch.Tensor):
        self.nF += int(ferr.numel())
        self.seF += float(torch.sum(ferr * ferr).detach().cpu())

    def e_rmse(self):
        return math.sqrt(self.seE / max(self.nE, 1))

    def f_rmse(self):
        return math.sqrt(self.seF / max(self.nF, 1))


# ============================================================
# TRAINING
# ============================================================

def train():
    set_seed(SEED)

    print("========================================================")
    print(" KAN-EAM Ag Stage 1 Training")
    print("========================================================")
    print(f"[INFO] Device: {DEVICE}, dtype={DTYPE}")
    print(f"[INFO] DATA_ROOT: {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT:  {OUT_ROOT}")

    all_z = scan_species_from_npz_files(DATA_ROOT)
    sp = make_species_info(all_z)

    print(f"[INFO] Elements Z: {[sp.type_to_z[i] for i in range(sp.n_types)]}")
    print(f"[INFO] Number of species: {sp.n_types}")

    if sp.n_types != 1 or list(sp.z_to_type.keys())[0] != 47:
        print("[WARN] This script is designed for single-element Ag, Z=47.")
        print(f"[WARN] Detected species: {sp.z_to_type}")

    ds_all = FrameDataset(DATA_ROOT, sp)

    print(f"[INFO] Found {len(ds_all)} .npz files")

    e_scale, f_scale = estimate_scales(ds_all, n_sample=200)

    print(
        f"[INFO] Estimated scales: "
        f"e_scale={e_scale:.6e} eV/atom, "
        f"f_scale={f_scale:.6e} eV/Angstrom"
    )

    try:
        rh_min0, rh_mean0, rh_max0 = estimate_rhobar_range(ds_all, sp, n_sample=30)

        print(
            f"[INFO] Initial rhobar estimate: "
            f"min={rh_min0:.3f}, mean={rh_mean0:.3f}, max={rh_max0:.3f}, "
            f"RHOMAX={RHOMAX:.3f}"
        )

    except Exception as e:
        print(f"[WARN] Could not estimate initial rhobar range: {e}")

    # ------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------
    idxs = list(range(len(ds_all)))
    random.shuffle(idxs)

    n_train = int(TRAIN_FRAC * len(idxs))

    train_ds = torch.utils.data.Subset(ds_all, idxs[:n_train])
    val_ds = torch.utils.data.Subset(ds_all, idxs[n_train:])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_one,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_one,
    )

    print(
        f"[INFO] Frames: total={len(ds_all)}, "
        f"train={len(train_ds)}, val={len(val_ds)}"
    )

    # ------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------
    model = KANEAM_Cubic(sp).to(DEVICE).to(DTYPE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1.0e-5,
    )

    best_balanced = float("inf")

    best_ckpt = CKPT_DIR / "BEST_BALANCED.pt"
    last_ckpt = CKPT_DIR / "LAST.pt"

    history_plot_dir = PLOT_DIR / "history"
    history_npz_path = OUT_ROOT / "training_history.npz"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_e_rmse": [],
        "val_e_rmse": [],
        "train_f_rmse": [],
        "val_f_rmse": [],
        "balanced_score": [],
        "lr": [],
    }

    best_stop_metric = float("inf")
    best_stop_epoch = 0
    no_improve_count = 0

    def stage_weights(ep: int):
        if ep <= int(0.60 * EPOCHS):
            return W_F_STAGE1, W_REG_STAGE1
        elif ep <= int(0.85 * EPOCHS):
            return W_F_STAGE2, W_REG_STAGE2
        else:
            return W_F_STAGE3, W_REG_STAGE3

    def run_epoch(loader, train_mode: bool, ep: int):
        model.train(train_mode)

        run = Running()
        total_loss = 0.0
        rhobar_maxs = []

        if train_mode:
            opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(loader):
            cell = torch.tensor(batch["cell"], device=DEVICE, dtype=DTYPE)
            pbc = torch.tensor(batch["pbc"], device=DEVICE)
            pos = torch.tensor(
                batch["pos"],
                device=DEVICE,
                dtype=DTYPE,
                requires_grad=True,
            )
            types = torch.tensor(batch["types"], device=DEVICE, dtype=torch.long)

            E_true = torch.tensor(batch["energy"], device=DEVICE, dtype=DTYPE)

            if batch["forces"] is not None:
                F_true = torch.tensor(batch["forces"], device=DEVICE, dtype=DTYPE)
            else:
                F_true = None

            nat = batch["natoms"]

            if (
                PRINT_RHOBAR_EVERY > 0
                and ep % PRINT_RHOBAR_EVERY == 0
                and it == 0
            ):
                E_pred, F_pred, rh = model.forward_energy_forces(
                    cell,
                    pbc,
                    pos,
                    types,
                    return_rhobar=True,
                )

                rhobar_maxs.append(float(rh.max().cpu()))

            else:
                E_pred, F_pred = model.forward_energy_forces(
                    cell,
                    pbc,
                    pos,
                    types,
                    return_rhobar=False,
                )

            # ------------------------------------------------------------
            # Anchor weights
            # ------------------------------------------------------------
            wE_anchor, wF_anchor = frame_anchor_weights(batch["path"])

            # ------------------------------------------------------------
            # Energy loss
            # Use energy per atom error.
            # ------------------------------------------------------------
            e_err = ((E_pred - E_true) / nat) / e_scale
            loss_E = wE_anchor * e_err * e_err

            # ------------------------------------------------------------
            # Force loss
            # ------------------------------------------------------------
            if F_true is not None:
                f_err = (F_pred - F_true) / f_scale

                if USE_HUBER_FORCES:
                    beta_norm = FORCE_HUBER_BETA / (f_scale + 1.0e-12)
                    loss_F_raw = F.smooth_l1_loss(
                        f_err,
                        torch.zeros_like(f_err),
                        beta=beta_norm,
                    )
                else:
                    loss_F_raw = torch.mean(f_err * f_err)

                loss_F = wF_anchor * loss_F_raw

            else:
                f_err = None
                loss_F = torch.zeros((), device=DEVICE, dtype=DTYPE)

            # ------------------------------------------------------------
            # Regularization
            # For the first few epochs, do not regularize F too strongly.
            # ------------------------------------------------------------
            W_F, W_REG = stage_weights(ep)

            if ep <= FREEZE_F_EPOCHS:
                reg = torch.zeros((), device=DEVICE, dtype=DTYPE)

                for s in model.rho:
                    reg = reg + LAMBDA_SMOOTH_RHO * s.smoothness_penalty()
                    reg = reg + LAMBDA_MONO_RHO * s.monotone_decreasing_penalty(
                        n_grid=MONO_GRID
                    )

                    p_val, p_slope = s.right_value_and_slope_penalty(0.0, 0.0)

                    reg = reg + LAMBDA_RHO_RC_VAL * p_val
                    reg = reg + LAMBDA_RHO_RC_SLOPE * p_slope

                for s in model.phi:
                    reg = reg + LAMBDA_SMOOTH_PHI * s.smoothness_penalty()

                    p_val, p_slope = s.right_value_and_slope_penalty(0.0, 0.0)

                    reg = reg + LAMBDA_PHI_RC_VAL * p_val
                    reg = reg + LAMBDA_PHI_RC_SLOPE * p_slope

            else:
                reg = model.regularization()

            loss = W_E * loss_E + W_F * loss_F + W_REG * reg

            if train_mode:
                (loss / GRAD_ACCUM_STEPS).backward()

                if (it + 1) % GRAD_ACCUM_STEPS == 0:
                    if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            CLIP_GRAD_NORM,
                        )

                    opt.step()
                    opt.zero_grad(set_to_none=True)

            run.add_energy(e_err.detach())

            if f_err is not None:
                run.add_forces(f_err.detach())

            total_loss += float(loss.detach().cpu())

        if train_mode:
            has_grad = any(p.grad is not None for p in model.parameters())

            if has_grad:
                if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        CLIP_GRAD_NORM,
                    )

                opt.step()
                opt.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(len(loader), 1)

        e_rmse_phys = run.e_rmse() * e_scale
        f_rmse_phys = run.f_rmse() * f_scale

        max_rh = max(rhobar_maxs) if len(rhobar_maxs) else None

        return avg_loss, e_rmse_phys, f_rmse_phys, max_rh

    def save_snapshot(
        tag_prefix: str,
        ep: int,
        ckpt_path: Path,
        ckpt_payload: dict,
    ):
        torch.save(ckpt_payload, ckpt_path)

        export_path = EXPORT_DIR / f"KAN_EAM_{tag_prefix}_epoch{ep:04d}.npz"

        model.export_all(export_path)

        plot_export_npz(
            export_path,
            PLOT_DIR / f"{tag_prefix}_epoch{ep:04d}",
            tag=f"{tag_prefix}_epoch{ep:04d}",
            dump_all=True,
        )

        print(f"[SAVE] {tag_prefix} checkpoint: {ckpt_path}")
        print(f"[SAVE] {tag_prefix} exported functions: {export_path}")

    # ------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_e, tr_f, tr_rhmax = run_epoch(train_loader, True, ep)
        va_loss, va_e, va_f, va_rhmax = run_epoch(val_loader, False, ep)

        dt = time.time() - t0

        lr_now = opt.param_groups[0]["lr"]

        W_F_now, W_REG_now = stage_weights(ep)

        balanced_score = (va_e / E_REF) + ALPHA_BAL * (va_f / F_REF)

        print(
            f"[Epoch {ep:04d}/{EPOCHS}] "
            f"lr={lr_now:.2e}, wF={W_F_now:.2f}, wReg={W_REG_now:.2f} | "
            f"train loss={tr_loss:.6e}, "
            f"E_RMSE={tr_e:.6e} eV/atom, "
            f"F_RMSE={tr_f:.6e} eV/A | "
            f"val loss={va_loss:.6e}, "
            f"E_RMSE={va_e:.6e} eV/atom, "
            f"F_RMSE={va_f:.6e} eV/A, "
            f"BAL={balanced_score:.6f} | "
            f"time={dt:.1f}s"
        )

        if tr_rhmax is not None:
            print(f"  [RHOBAR] train max/RHOMAX = {tr_rhmax / RHOMAX:.3f}")

            if tr_rhmax / RHOMAX > RHO_WARN_FRAC:
                print(
                    f"  [WARN] rhobar is approaching RHOMAX={RHOMAX}. "
                    f"Consider increasing RHOMAX."
                )

        scheduler.step(va_loss)

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_e_rmse"].append(tr_e)
        history["val_e_rmse"].append(va_e)
        history["train_f_rmse"].append(tr_f)
        history["val_f_rmse"].append(va_f)
        history["balanced_score"].append(balanced_score)
        history["lr"].append(lr_now)

        plot_metric_history(history, history_plot_dir)
        save_history_npz(history, history_npz_path)

        ckpt_payload = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "species": model.sp.__dict__,
            "scales": {
                "e_scale": e_scale,
                "f_scale": f_scale,
            },
            "settings": {
                "RCUT": RCUT,
                "RMIN": RMIN,
                "RHOMIN": RHOMIN,
                "RHOMAX": RHOMAX,
                "N_CTRL_R": N_CTRL_R,
                "N_CTRL_F": N_CTRL_F,
                "USE_ZBL": USE_ZBL,
                "ZBL_R_SWITCH": ZBL_R_SWITCH,
            },
            "metrics": {
                "train_loss": tr_loss,
                "train_e_rmse": tr_e,
                "train_f_rmse": tr_f,
                "val_loss": va_loss,
                "val_e_rmse": va_e,
                "val_f_rmse": va_f,
                "balanced_score": balanced_score,
            },
        }

        torch.save(ckpt_payload, last_ckpt)

        if balanced_score < best_balanced:
            best_balanced = balanced_score
            save_snapshot("BEST_BALANCED", ep, best_ckpt, ckpt_payload)

        stop_metric = balanced_score

        improved = (best_stop_metric - stop_metric) > EARLY_STOPPING_MIN_DELTA

        if improved:
            best_stop_metric = stop_metric
            best_stop_epoch = ep
            no_improve_count = 0
        else:
            no_improve_count += 1

        print(
            f"[EARLY-STOP] current={stop_metric:.6e}, "
            f"best={best_stop_metric:.6e}, "
            f"best_epoch={best_stop_epoch}, "
            f"patience={no_improve_count}/{EARLY_STOPPING_PATIENCE}"
        )

        if (
            EARLY_STOPPING_ENABLED
            and ep >= MIN_EPOCHS_BEFORE_STOP
            and no_improve_count >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"[STOP] Early stopping at epoch {ep}. "
                f"Best epoch was {best_stop_epoch}."
            )
            break

    # ------------------------------------------------------------
    # Final export
    # ------------------------------------------------------------
    final_export = EXPORT_DIR / "KAN_EAM_FINAL.npz"
    model.export_all(final_export)

    plot_export_npz(
        final_export,
        PLOT_DIR / "FINAL",
        tag="FINAL",
        dump_all=True,
    )

    print("[DONE] Training finished.")
    print(f"[DONE] Outputs written to: {OUT_ROOT}")
    print(f"[DONE] Final exported functions: {final_export}")


if __name__ == "__main__":
    train()