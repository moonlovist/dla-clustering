#!/usr/bin/env python3
"""Measure the anisotropic 3D DLA x QSO cross-correlation.

Both samples require their own random catalog.  pycorr evaluates the cross
Landy--Szalay estimator

    (D_DLA D_QSO - D_DLA R_QSO - R_DLA D_QSO + R_DLA R_QSO) / (R_DLA R_QSO).

The DLA random should normally be made with the exact-QSO-LOS generator.
"""

from __future__ import annotations

import argparse
import os
import time

import clustering_3d_lib as L


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dla-file", required=True)
    parser.add_argument("--dla-random-file", required=True)
    parser.add_argument("--qso-file", required=True)
    parser.add_argument("--qso-random-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default=None)
    parser.add_argument("--dla-z-column", default="Z_DLA_RSD")
    parser.add_argument("--dla-random-z-column", default="Z_DLA_RSD")
    parser.add_argument("--qso-z-column", default="Z_QSO_RSD")
    parser.add_argument("--qso-random-z-column", default="Z_QSO_RSD")
    parser.add_argument("--weight-column", default="WEIGHT")
    parser.add_argument("--h", type=float, default=0.7)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--s-min", type=float, default=20.0)
    parser.add_argument("--s-max", type=float, default=200.0)
    parser.add_argument("--s-step", type=float, default=4.0)
    parser.add_argument("--n-mu", type=int, default=120)
    parser.add_argument("--ells", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument(
        "--los", choices=("midpoint", "endpoint", "firstpoint", "x", "y", "z"),
        default="midpoint",
    )
    parser.add_argument("--nthreads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    parser.add_argument("--dla-random-fraction", type=float, default=1.0)
    parser.add_argument("--qso-random-fraction", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.check_mpi_launch()
    from pycorr import TwoPointCorrelationFunction

    s_edges, mu_edges = L.make_edges(args.s_min, args.s_max, args.s_step, args.n_mu)
    started = time.time()
    if L.RANK == 0:
        specifications = [
            (args.dla_file, args.dla_z_column, "DLA data"),
            (args.dla_random_file, args.dla_random_z_column, "DLA random"),
            (args.qso_file, args.qso_z_column, "QSO data"),
            (args.qso_random_file, args.qso_random_z_column, "QSO random"),
        ]
        catalogs = [L.read_catalog(path, zcol, args.weight_column, label=label) for path, zcol, label in specifications]
        fractions = [args.test_fraction, args.test_fraction * args.dla_random_fraction,
                     args.test_fraction, args.test_fraction * args.qso_random_fraction]
        catalogs = [L.thin_catalog(cat, fraction, args.seed + i) for i, (cat, fraction) in enumerate(zip(catalogs, fractions))]
        positions = [L.radec_z_to_xyz(cat, args.h, args.omega_m) for cat in catalogs]
        weights = [cat["w"] for cat in catalogs]
        counts = dict(zip(("dla", "dla_random", "qso", "qso_random"),
                          [len(cat["ra"]) for cat in catalogs]))
    else:
        positions = [None] * 4
        weights = [None] * 4
        counts = None

    L.log(f"DLA x QSO: MPI ranks={L.SIZE}; threads/rank={args.nthreads}")
    pair_started = time.time()
    result = TwoPointCorrelationFunction(
        "smu",
        (s_edges, mu_edges),
        data_positions1=positions[0], randoms_positions1=positions[1],
        data_positions2=positions[2], randoms_positions2=positions[3],
        data_weights1=weights[0], randoms_weights1=weights[1],
        data_weights2=weights[2], randoms_weights2=weights[3],
        estimator="landyszalay",
        engine="corrfunc",
        nthreads=args.nthreads,
        position_type="xyz",
        los=args.los,
        compute_sepsavg=True,
        mpicomm=L.COMM if L.SIZE > 1 else None,
        mpiroot=0 if L.SIZE > 1 else None,
    )
    pair_seconds = time.time() - pair_started
    if L.RANK != 0:
        return

    ells = tuple(args.ells)
    separation, poles = L.unpack_poles(result, ells)
    metadata = {
        "method": "pycorr_cross_landy_szalay",
        "files": {
            "dla": args.dla_file, "dla_random": args.dla_random_file,
            "qso": args.qso_file, "qso_random": args.qso_random_file,
            "output": args.output,
        },
        "columns": {
            "dla_z": args.dla_z_column, "dla_random_z": args.dla_random_z_column,
            "qso_z": args.qso_z_column, "qso_random_z": args.qso_random_z_column,
            "weight": args.weight_column,
        },
        "counts": counts,
        "cosmology": {"h": args.h, "omega_m": args.omega_m},
        "binning": {
            "s_min": args.s_min, "s_max": args.s_max, "s_step": args.s_step,
            "n_mu": args.n_mu, "ells": args.ells, "los": args.los,
        },
        "parallelization": {"mpi_ranks": L.SIZE, "threads_per_rank": args.nthreads},
        "fractions": {
            "test": args.test_fraction,
            "dla_random": args.dla_random_fraction,
            "qso_random": args.qso_random_fraction,
        },
        "timing": {"pair_seconds": pair_seconds, "total_seconds": time.time() - started},
    }
    L.save_measurement(
        args.output, separation, poles, ells, s_edges, mu_edges, metadata, args.meta_output
    )


if __name__ == "__main__":
    main()
