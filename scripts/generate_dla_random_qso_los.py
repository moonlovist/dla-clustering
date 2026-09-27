#!/usr/bin/env python3
"""Generate a DLA random catalog on the actual QSO sightlines.

Redshifts are drawn first from the reference DLA n(z).  Each draw is then
assigned uniformly to a QSO sightline on which that absorber could be observed.
This preserves the empirical radial selection without giving long-path QSOs an
extra weight twice.  Three window modes are available:

``none``   reproduces a balanced angular-only QSO-LOS catalog;
``zqso``   requires only z_DLA < z_QSO;
``forest`` applies the per-QSO forest, blue-cutoff, and proximity window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from clustering_3d_lib import dla_forest_window, require_columns, sample_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--qso-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default=None)
    parser.add_argument("--reference-hdu", type=int, default=1)
    parser.add_argument("--qso-hdu", type=int, default=1)
    parser.add_argument("--factor", type=float, default=20.0)
    parser.add_argument("--n-random", type=int, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--nest", action="store_true")
    parser.add_argument("--z-columns", nargs="+", default=["Z_DLA_RSD"])
    parser.add_argument("--primary-z-column", default="Z_DLA_RSD")
    parser.add_argument(
        "--z-method", choices=("histogram", "empirical-row"), default="empirical-row"
    )
    parser.add_argument("--z-bins", type=int, default=200)
    parser.add_argument("--qso-ra-column", default="RA")
    parser.add_argument("--qso-dec-column", default="DEC")
    parser.add_argument("--qso-z-column", default="Z_QSO_NO_RSD")
    parser.add_argument("--qso-id-column", default="MOCKID")
    parser.add_argument("--window", choices=("none", "zqso", "forest"), default="forest")
    parser.add_argument("--observed-min", type=float, default=2550.0)
    parser.add_argument("--forest-rest-min", type=float, default=911.0)
    parser.add_argument("--forest-rest-max", type=float, default=1216.0)
    parser.add_argument("--proximity-kms", type=float, default=3000.0)
    parser.add_argument(
        "--assignment-bins", type=int, default=500,
        help="Redshift batches used to find eligible sightlines efficiently.",
    )
    return parser.parse_args()


def unique_sightlines(table: Table, args: argparse.Namespace) -> Table:
    needed = [args.qso_ra_column, args.qso_dec_column]
    if args.window != "none":
        needed.append(args.qso_z_column)
    require_columns(table, needed, "QSO catalog")
    keep = np.isfinite(np.asarray(table[args.qso_ra_column], float))
    keep &= np.isfinite(np.asarray(table[args.qso_dec_column], float))
    if args.window != "none":
        keep &= np.isfinite(np.asarray(table[args.qso_z_column], float))
    table = table[keep]
    if args.qso_id_column in table.colnames:
        _, index = np.unique(np.asarray(table[args.qso_id_column]), return_index=True)
    else:
        coordinates = np.rec.fromarrays(
            [np.asarray(table[args.qso_ra_column], float), np.asarray(table[args.qso_dec_column], float)]
        )
        _, index = np.unique(coordinates, return_index=True)
    return table[np.sort(index)]


def assign_balanced(n_los: int, n_output: int, rng: np.random.Generator) -> np.ndarray:
    full, remainder = divmod(n_output, n_los)
    index = np.tile(np.arange(n_los, dtype="i8"), full)
    if remainder:
        index = np.concatenate([index, rng.choice(n_los, remainder, replace=False)])
    rng.shuffle(index)
    return index


def assign_eligible(
    redshift: np.ndarray,
    z_low: np.ndarray,
    z_high: np.ndarray,
    n_batches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign every redshift uniformly among its exactly eligible sightlines."""
    edges = np.linspace(redshift.min(), redshift.max(), n_batches + 1)
    group = np.clip(np.searchsorted(edges, redshift, side="right") - 1, 0, n_batches - 1)
    assigned = np.empty(len(redshift), dtype="i8")
    for number in np.unique(group):
        rows = np.flatnonzero(group == number)
        candidates = np.flatnonzero((z_low < edges[number + 1]) & (z_high > edges[number]))
        if not len(candidates):
            raise ValueError(f"No QSO sightline overlaps redshift bin {number}.")
        choice = rng.choice(candidates, size=len(rows), replace=True)
        valid = (z_low[choice] <= redshift[rows]) & (redshift[rows] < z_high[choice])
        attempts = 0
        while not valid.all() and attempts < 100:
            choice[~valid] = rng.choice(candidates, size=(~valid).sum(), replace=True)
            valid = (z_low[choice] <= redshift[rows]) & (redshift[rows] < z_high[choice])
            attempts += 1
        if not valid.all():
            for local in np.flatnonzero(~valid):
                exact = np.flatnonzero(
                    (z_low <= redshift[rows[local]]) & (redshift[rows[local]] < z_high)
                )
                if not len(exact):
                    raise ValueError(f"No eligible QSO for z_DLA={redshift[rows[local]]:.6f}.")
                choice[local] = rng.choice(exact)
        assigned[rows] = choice
    return assigned


def main() -> None:
    args = parse_args()
    import healpy as hp

    if args.primary_z_column not in args.z_columns:
        raise ValueError("--primary-z-column must be included in --z-columns.")
    if args.factor <= 0 or args.z_bins < 2 or args.assignment_bins < 2:
        raise ValueError("factor, z-bins, and assignment-bins must be positive.")
    if not hp.isnsideok(args.nside):
        raise ValueError(f"Invalid HEALPix NSIDE={args.nside}.")

    started = time.time()
    rng = np.random.default_rng(args.seed)
    reference = Table.read(args.reference_file, hdu=args.reference_hdu)
    require_columns(reference, args.z_columns, "DLA reference")
    finite = np.ones(len(reference), dtype=bool)
    for column in args.z_columns:
        finite &= np.isfinite(np.asarray(reference[column], float))
    reference = reference[finite]
    qso = unique_sightlines(Table.read(args.qso_file, hdu=args.qso_hdu), args)
    if not len(reference) or not len(qso):
        raise ValueError("The reference DLA or QSO sightline catalog is empty.")

    n_random = args.n_random or int(round(args.factor * len(reference)))
    source_rows = rng.integers(0, len(reference), size=n_random)
    if args.z_method == "empirical-row":
        redshifts = {
            column: np.asarray(reference[column][source_rows], dtype="f4")
            for column in args.z_columns
        }
    else:
        primary_draw = sample_histogram(
            reference[args.primary_z_column], n_random, args.z_bins, rng
        )
        redshifts = {args.primary_z_column: primary_draw.astype("f4")}
        for column in args.z_columns:
            if column == args.primary_z_column:
                continue
            delta = np.asarray(
                reference[column] - reference[args.primary_z_column], dtype=float
            )
            redshifts[column] = (primary_draw + delta[source_rows]).astype("f4")
    primary = np.asarray(redshifts[args.primary_z_column], dtype=float)

    if args.window == "none":
        los_index = assign_balanced(len(qso), n_random, rng)
        z_low = z_high = None
    else:
        z_qso = np.asarray(qso[args.qso_z_column], dtype=float)
        if args.window == "zqso":
            z_low = np.full(len(qso), -np.inf)
            z_high = z_qso
        else:
            z_low, z_high = dla_forest_window(
                z_qso, args.observed_min, args.forest_rest_min,
                args.forest_rest_max, args.proximity_kms,
            )
        usable = z_high > z_low
        qso = qso[usable]
        z_low, z_high = z_low[usable], z_high[usable]
        los_index = assign_eligible(primary, z_low, z_high, args.assignment_bins, rng)

    ra = np.asarray(qso[args.qso_ra_column][los_index], dtype="f4") % 360.0
    dec = np.asarray(qso[args.qso_dec_column][los_index], dtype="f4")
    pixels = hp.ang2pix(args.nside, ra, dec, lonlat=True, nest=args.nest).astype("i4")
    output = Table({"RA": ra, "DEC": dec, "PIXNUM": pixels})
    for column, values in redshifts.items():
        output[column] = values
    if args.qso_z_column in qso.colnames:
        output[args.qso_z_column] = np.asarray(qso[args.qso_z_column][los_index])
    if args.qso_id_column in qso.colnames:
        output[args.qso_id_column] = np.asarray(qso[args.qso_id_column][los_index])

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.write(path, overwrite=True)
    counts = np.bincount(los_index, minlength=len(qso))
    meta = {
        "method": "exact_qso_sightlines_plus_empirical_nz",
        "reference_file": args.reference_file,
        "qso_file": args.qso_file,
        "output_file": str(path),
        "window": args.window,
        "window_parameters": {
            "observed_min": args.observed_min,
            "forest_rest_min": args.forest_rest_min,
            "forest_rest_max": args.forest_rest_max,
            "proximity_kms": args.proximity_kms,
        },
        "n_reference": len(reference),
        "n_unique_usable_qso": len(qso),
        "n_random": n_random,
        "n_used_sightlines": int(np.count_nonzero(counts)),
        "z_columns": args.z_columns,
        "primary_z_column": args.primary_z_column,
        "z_method": args.z_method,
        "seed": args.seed,
        "nside": args.nside,
        "nest": args.nest,
        "runtime_seconds": time.time() - started,
    }
    sidecar = Path(args.meta_output) if args.meta_output else Path(str(path) + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)
    print(f"Saved {path}\nSaved {sidecar}")


if __name__ == "__main__":
    main()
