from __future__ import annotations

import numpy as np

from solarvision.features import FEATURE_NAMES, extract, extract_matrix
from solarvision.generate import render
from solarvision.labels import DefectClass
from tests.conftest import make_panel


def test_feature_vector_shape_and_names():
    f = extract(make_panel(DefectClass.HEALTHY, seed=1))
    assert f.vector().shape == (len(FEATURE_NAMES),)
    assert set(f.as_dict()) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 14


def test_hotspot_shows_a_bright_blob():
    healthy = extract(make_panel(DefectClass.HEALTHY, seed=1))
    hot = extract(make_panel(DefectClass.HOTSPOT, seed=2))
    # A clear hotspot stands far above the background and much higher than healthy.
    assert hot.bright_blob_contrast > healthy.bright_blob_contrast + 2.0
    assert hot.bright_blob_area > 0.0
    assert hot.bright_tail > healthy.bright_tail


def test_crack_is_thin_and_elongated():
    crack = extract(make_panel(DefectClass.CRACK, seed=3))
    dead = extract(make_panel(DefectClass.DEAD_CELL, seed=4))
    # A crack's dark region is long and thin; a dead cell's is blocky.
    assert crack.dark_blob_elongation > dead.dark_blob_elongation
    assert crack.dark_blob_fill < dead.dark_blob_fill


def test_dead_cell_is_a_large_solid_dark_block():
    dead = extract(make_panel(DefectClass.DEAD_CELL, seed=4))
    healthy = extract(make_panel(DefectClass.HEALTHY, seed=1))
    assert dead.dark_blob_area > healthy.dark_blob_area
    assert dead.dark_blob_area > 0.02  # occupies a meaningful chunk of the panel
    assert dead.dark_blob_fill > 0.5  # fills its bounding box


def test_crack_raises_edge_density_above_healthy():
    # A crack is a high-gradient streak, so it adds edge energy a healthy panel
    # (whose only edges are the faint cell grid) does not have.
    healthy = extract(make_panel(DefectClass.HEALTHY, seed=1))
    for seed in (3, 13, 23):
        rng = np.random.default_rng(seed)
        crack = extract(render(DefectClass.CRACK, rng, severity=0.9, size_frac=0.9))
        assert crack.edge_density > healthy.edge_density
        assert crack.mean_gradient > healthy.mean_gradient


def test_extract_matrix_stacks_rows():
    imgs = [make_panel(DefectClass.HEALTHY, seed=1), make_panel(DefectClass.HOTSPOT, seed=2)]
    matrix = extract_matrix(imgs)
    assert matrix.shape == (2, len(FEATURE_NAMES))
