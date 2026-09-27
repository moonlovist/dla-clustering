"""Unit tests for the dependency-light parts of the 3D random generators."""

import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import clustering_3d_lib as L
import generate_dla_random_qso_los as Q


def test_forest_window_is_ordered_and_below_qso():
    z_qso = np.array([2.5, 3.0, 3.5])
    z_low, z_high = L.dla_forest_window(
        z_qso,
        observed_min=3600.0,
        rest_min=911.0,
        rest_max=1216.0,
        proximity_kms=3000.0,
    )
    assert np.all(z_low < z_high)
    assert np.all(z_high < z_qso)
    assert np.all(z_low >= 3600.0 / L.LYA - 1.0)


def test_balanced_assignment_differs_by_at_most_one():
    index = Q.assign_balanced(17, 103, np.random.default_rng(2))
    counts = np.bincount(index, minlength=17)
    assert len(index) == 103
    assert counts.max() - counts.min() <= 1


def test_eligible_assignment_obeys_each_exact_window():
    redshift = np.linspace(2.05, 2.70, 1000)
    z_low = np.array([2.0, 2.2, 2.4])
    z_high = np.array([2.4, 2.6, 2.8])
    index = Q.assign_eligible(
        redshift, z_low, z_high, n_batches=40, rng=np.random.default_rng(7)
    )
    assert np.all(z_low[index] <= redshift)
    assert np.all(redshift < z_high[index])


def test_histogram_sampler_is_reproducible_and_in_range():
    values = np.linspace(1.8, 3.2, 101)
    first = L.sample_histogram(values, 5000, 20, np.random.default_rng(11))
    second = L.sample_histogram(values, 5000, 20, np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert first.min() >= values.min()
    assert first.max() <= values.max()
