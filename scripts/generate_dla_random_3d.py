#!/usr/bin/env python3
"""Generate the 3D-mask control random catalog for DLA clustering.

Angular positions are uniform within the HEALPix pixels occupied by a reference
DLA catalog.  Redshifts follow its empirical global n(z).  This construction
treats DLAs as ordinary 3D tracers; it does *not* preserve the discrete QSO
sightlines on which absorbers are observed.  Use the QSO-LOS generator for the
fiducial measurement and this script as a controlled comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from clustering_3d_lib import require_columns, sample_histogram


hp = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default=None)
    parser.add_argument("--hdu", type=int, default=1)
    parser.add_argument("--factor", type=float, default=20.0)
    parser.add_argument("--n-random", type=int, default=None)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--nest", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--z-columns", nargs="+", default=["Z_DLA_RSD"])
    parser.add_argument(
        "--z-method", choices=("histogram", "empirical-row"), default="histogram"
    )
    parser.add_argument("--z-bins", type=int, default=100)
    parser.add_argument("--proposal-size", type=int, default=1_000_000)
    return parser.parse_args()


def perpendicular_basis(centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros_like(centres)
    reference[:, 2] = 1.0
    reference[np.abs(centres[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    first = np.cross(reference, centres)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return first, np.cross(centres, first)


def sample_occupied_pixels(
    occupied: np.ndarray,
    n_output: int,
    nside: int,
    nest: bool,
    proposal_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Draw exactly uniform directions in a union of equal-area HEALPix cells."""
    ra = np.empty(n_output, dtype="f4")
    dec = np.empty(n_output, dtype="f4")
    pixels = np.empty(n_output, dtype="i4")
    cap = hp.max_pixrad(nside) * 1.000001
    cos_cap = np.cos(cap)
    filled = proposed = 0

    while filled < n_output:
        n_try = min(proposal_size, max(20_000, 2 * (n_output - filled)))
        selected = rng.choice(occupied, size=n_try)
        centres = np.column_stack(hp.pix2vec(nside, selected, nest=nest))
        first, second = perpendicular_basis(centres)
        cos_angle = rng.uniform(cos_cap, 1.0, n_try)
        sin_angle = np.sqrt(1.0 - cos_angle**2)
        azimuth = rng.uniform(0.0, 2.0 * np.pi, n_try)
        tangent = np.cos(azimuth)[:, None] * first + np.sin(azimuth)[:, None] * second
        vectors = cos_angle[:, None] * centres + sin_angle[:, None] * tangent
        proposed_pixel = hp.vec2pix(nside, *vectors.T, nest=nest)
        keep = proposed_pixel == selected
        vectors = vectors[keep]
        selected = selected[keep]
        take = min(len(selected), n_output - filled)
        if take:
            theta, phi = hp.vec2ang(vectors[:take])
            target = slice(filled, filled + take)
            ra[target] = np.rad2deg(phi)
            dec[target] = 90.0 - np.rad2deg(theta)
            pixels[target] = selected[:take]
            filled += take
        proposed += n_try
    return ra, dec, pixels, n_output / proposed


def main() -> None:
    global hp
    args = parse_args()
    import healpy as hp_module

    hp = hp_module
    if args.factor <= 0 or args.z_bins < 2 or args.proposal_size <= 0:
        raise ValueError("factor, z-bins, and proposal-size must be positive.")
    if not hp.isnsideok(args.nside):
        raise ValueError(f"Invalid HEALPix NSIDE={args.nside}.")

    started = time.time()
    rng = np.random.default_rng(args.seed)
    reference = Table.read(args.reference_file, hdu=args.hdu)
    require_columns(reference, ["RA", "DEC", *args.z_columns], "DLA reference")
    finite = np.ones(len(reference), dtype=bool)
    for column in ["RA", "DEC", *args.z_columns]:
        finite &= np.isfinite(np.asarray(reference[column], dtype=float))
    reference = reference[finite]
    if not len(reference):
        raise ValueError("No finite reference rows remain.")

    n_random = args.n_random or int(round(args.factor * len(reference)))
    theta = np.deg2rad(90.0 - np.asarray(reference["DEC"], dtype=float))
    phi = np.deg2rad(np.asarray(reference["RA"], dtype=float) % 360.0)
    occupied = np.unique(hp.ang2pix(args.nside, theta, phi, nest=args.nest))
    print(f"DLA reference={len(reference):,}; random={n_random:,}; pixels={len(occupied):,}")

    ra, dec, pixels, acceptance = sample_occupied_pixels(
        occupied, n_random, args.nside, args.nest, args.proposal_size, rng
    )
    output = Table({"RA": ra, "DEC": dec, "PIXNUM": pixels})
    source_rows = rng.integers(0, len(reference), size=n_random)
    if args.z_method == "empirical-row":
        for column in args.z_columns:
            output[column] = np.asarray(reference[column][source_rows], dtype="f4")
    else:
        primary = args.z_columns[0]
        primary_draw = sample_histogram(reference[primary], n_random, args.z_bins, rng)
        output[primary] = primary_draw.astype("f4")
        # Preserve the measured RSD/no-RSD offsets instead of drawing correlated
        # redshift columns independently.
        for column in args.z_columns[1:]:
            delta = np.asarray(reference[column] - reference[primary], dtype=float)
            output[column] = (primary_draw + delta[source_rows]).astype("f4")

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.write(path, overwrite=True)
    meta = {
        "method": "uniform_occupied_healpix_plus_global_nz",
        "reference_file": args.reference_file,
        "output_file": str(path),
        "n_reference": len(reference),
        "n_random": n_random,
        "nside": args.nside,
        "nest": args.nest,
        "n_occupied_pixels": len(occupied),
        "z_columns": args.z_columns,
        "z_method": args.z_method,
        "z_bins": args.z_bins,
        "seed": args.seed,
        "angular_acceptance": acceptance,
        "scientific_role": "3D-mask control; does not preserve QSO sightlines",
        "runtime_seconds": time.time() - started,
    }
    sidecar = Path(args.meta_output) if args.meta_output else Path(str(path) + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Saved {path}\nSaved {sidecar}")


if __name__ == "__main__":
    main()
