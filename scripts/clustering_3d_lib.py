#!/usr/bin/env python3
"""Shared utilities for the transfer-ready 3D DLA clustering scripts.

The module intentionally contains only catalog I/O, coordinate conversion,
redshift sampling, MPI bookkeeping, and result serialization.  Survey-specific
selection cuts should be made before calling the measurement scripts, or be
encoded in the random catalogs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


C_KMS = 299792.458
LYA = 1215.67

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    MPI_IMPORT_ERROR = None
except ImportError as exc:  # serial execution remains supported
    COMM = None
    RANK = 0
    SIZE = 1
    MPI_IMPORT_ERROR = exc


def log(message: str) -> None:
    """Print once when a script is launched with MPI."""
    if RANK == 0:
        print(message, flush=True)


def check_mpi_launch() -> None:
    """Catch the common mistake of launching independent serial processes."""
    import os

    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ntasks > 1 and SIZE == 1:
        detail = "mpi4py is unavailable" if MPI_IMPORT_ERROR else "COMM_WORLD has size 1"
        raise RuntimeError(
            f"SLURM_NTASKS={ntasks}, but {detail}. This would start duplicate "
            "pair-count jobs. Use a working MPI launcher or one task with threads."
        )


def require_columns(table: Any, columns: list[str], label: str) -> None:
    missing = [name for name in columns if name not in table.colnames]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def read_catalog(
    path: str,
    z_column: str,
    weight_column: str | None = "WEIGHT",
    hdu: int = 1,
    label: str = "catalog",
) -> dict[str, np.ndarray | None]:
    """Read finite RA, DEC, redshift, and optional weights from a FITS table."""
    from astropy.table import Table

    table = Table.read(path, hdu=hdu)
    require_columns(table, ["RA", "DEC", z_column], label)
    ra = np.asarray(table["RA"], dtype="f8") % 360.0
    dec = np.asarray(table["DEC"], dtype="f8")
    redshift = np.asarray(table[z_column], dtype="f8")
    weight = None
    if weight_column and weight_column in table.colnames:
        weight = np.asarray(table[weight_column], dtype="f8")

    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(redshift)
    keep &= (dec >= -90.0) & (dec <= 90.0) & (redshift >= 0.0)
    if weight is not None:
        keep &= np.isfinite(weight) & (weight >= 0.0)
        weight = weight[keep]
        if np.allclose(weight, 1.0):
            weight = None

    result = {"ra": ra[keep], "dec": dec[keep], "z": redshift[keep], "w": weight}
    log(f"{label}: {len(ra):,} rows -> {keep.sum():,} usable; z column={z_column}")
    if not keep.any():
        raise ValueError(f"No usable rows remain in {label}.")
    return result


def radec_z_to_xyz(
    catalog: dict[str, np.ndarray | None], h: float, omega_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sky position and redshift to Cartesian comoving Mpc/h."""
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=100.0 * h, Om0=omega_m)
    redshift = np.asarray(catalog["z"])
    chi = cosmo.comoving_distance(redshift).value * h
    ra = np.deg2rad(np.asarray(catalog["ra"]))
    dec = np.deg2rad(np.asarray(catalog["dec"]))
    cdec = np.cos(dec)
    return (
        (chi * cdec * np.cos(ra)).astype("f4"),
        (chi * cdec * np.sin(ra)).astype("f4"),
        (chi * np.sin(dec)).astype("f4"),
    )


def thin_catalog(
    catalog: dict[str, np.ndarray | None], fraction: float, seed: int
) -> dict[str, np.ndarray | None]:
    """Reproducibly thin a catalog without changing its relative weights."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Catalog fraction must lie in (0, 1].")
    if fraction == 1.0:
        return catalog
    n_input = len(np.asarray(catalog["ra"]))
    n_keep = max(1, int(round(fraction * n_input)))
    index = np.random.default_rng(seed).choice(n_input, size=n_keep, replace=False)
    return {
        key: None if value is None else np.asarray(value)[index]
        for key, value in catalog.items()
    }


def sample_histogram(
    values: np.ndarray, n_output: int, bins: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw a smoothed empirical one-dimensional distribution."""
    values = np.asarray(values, dtype="f8")
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins)
    if not hist.sum():
        raise ValueError("Cannot sample an empty distribution.")
    chosen = rng.choice(len(hist), size=n_output, p=hist / hist.sum())
    return edges[chosen] + rng.random(n_output) * (edges[chosen + 1] - edges[chosen])


def dla_forest_window(
    z_qso: np.ndarray,
    observed_min: float,
    rest_min: float,
    rest_max: float,
    proximity_kms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the usable DLA-redshift interval for every QSO sightline."""
    z_qso = np.asarray(z_qso, dtype="f8")
    dz = (1.0 + z_qso) * proximity_kms / C_KMS
    z_low = np.maximum(
        observed_min / LYA - 1.0,
        (1.0 + z_qso) * rest_min / LYA - 1.0 + dz,
    )
    z_high = np.minimum(
        z_qso - dz,
        (1.0 + z_qso) * rest_max / LYA - 1.0 - dz,
    )
    return z_low, z_high


def make_edges(s_min: float, s_max: float, s_step: float, n_mu: int):
    if s_step <= 0.0 or s_max <= s_min or n_mu <= 0:
        raise ValueError("Require s_step>0, s_max>s_min, and n_mu>0.")
    return (
        np.arange(s_min, s_max + 0.5 * s_step, s_step),
        np.linspace(0.0, 1.0, n_mu + 1),
    )


def unpack_poles(result: Any, ells: tuple[int, ...]):
    output = result.get_corr(return_sep=True, mode="poles", ells=ells)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("Unexpected pycorr multipole output.")
    separation = np.asarray(output[0])
    poles = [np.asarray(value) for value in output[1]]
    return separation, poles


def save_measurement(
    output: str,
    separation: np.ndarray,
    poles: list[np.ndarray],
    ells: tuple[int, ...],
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    metadata: dict[str, Any],
    meta_output: str | None = None,
) -> None:
    """Write compact arrays plus a readable provenance sidecar."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "s": separation,
        "s_edges": s_edges,
        "mu_edges": mu_edges,
        "ells": np.asarray(ells, dtype="i4"),
    }
    arrays.update({f"xi{ell}": pole for ell, pole in zip(ells, poles)})
    np.savez(path, **arrays)

    sidecar = Path(meta_output) if meta_output else Path(str(path) + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    log(f"Saved {path}")
    log(f"Saved {sidecar}")
