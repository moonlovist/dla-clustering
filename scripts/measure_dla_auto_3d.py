#!/usr/bin/env python3
"""Measure the anisotropic 3D DLA auto-correlation.

The estimator is Landy--Szalay, evaluated by pycorr/Corrfunc in (s, mu) bins.
The saved NPZ contains the requested multipoles; the JSON sidecar records all
catalog, cosmology, binning, and parallelization choices.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import clustering_3d_lib as L


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dla-file", required=True)
    parser.add_argument("--random-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default=None)
    parser.add_argument("--data-hdu", type=int, default=1)
    parser.add_argument("--random-hdu", type=int, default=1)
    parser.add_argument("--data-z-column", default="Z_DLA_RSD")
    parser.add_argument("--random-z-column", default="Z_DLA_RSD")
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
    parser.add_argument("--random-fraction", type=float, default=1.0)
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
        data = L.read_catalog(
            args.dla_file, args.data_z_column, args.weight_column, args.data_hdu, "DLA data"
        )
        random = L.read_catalog(
            args.random_file, args.random_z_column, args.weight_column, args.random_hdu, "DLA random"
        )
        data = L.thin_catalog(data, args.test_fraction, args.seed)
        random = L.thin_catalog(random, args.test_fraction * args.random_fraction, args.seed + 1)
        positions_data = L.radec_z_to_xyz(data, args.h, args.omega_m)
        positions_random = L.radec_z_to_xyz(random, args.h, args.omega_m)
        weights_data, weights_random = data["w"], random["w"]
        counts = {"data": len(data["ra"]), "random": len(random["ra"])}
    else:
        positions_data = positions_random = None
        weights_data = weights_random = None
        counts = None

    L.log(f"DLA auto: MPI ranks={L.SIZE}; threads/rank={args.nthreads}")
    pair_started = time.time()
    result = TwoPointCorrelationFunction(
        "smu",
        (s_edges, mu_edges),
        data_positions1=positions_data,
        data_weights1=weights_data,
        randoms_positions1=positions_random,
        randoms_weights1=weights_random,
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
        "method": "pycorr_auto_landy_szalay",
        "files": {"data": args.dla_file, "random": args.random_file, "output": args.output},
        "columns": {
            "data_z": args.data_z_column,
            "random_z": args.random_z_column,
            "weight": args.weight_column,
        },
        "counts": counts,
        "cosmology": {"h": args.h, "omega_m": args.omega_m},
        "binning": {
            "s_min": args.s_min, "s_max": args.s_max, "s_step": args.s_step,
            "n_mu": args.n_mu, "ells": args.ells, "los": args.los,
        },
        "parallelization": {"mpi_ranks": L.SIZE, "threads_per_rank": args.nthreads},
        "fractions": {"test": args.test_fraction, "random": args.random_fraction},
        "timing": {"pair_seconds": pair_seconds, "total_seconds": time.time() - started},
    }
    L.save_measurement(
        args.output, separation, poles, ells, s_edges, mu_edges, metadata, args.meta_output
    )


if __name__ == "__main__":
    main()
