"""Tests for Bragg peak detection and 3D-ΔPDF."""

import numpy as np

from ndiff.analysis.bragg import BraggRemover, bragg_mask
from ndiff.analysis.bragg_fill import backfill_bragg
from ndiff.analysis.delta_pdf import _next_power_of_2, compute_delta_pdf
from ndiff.core import HKLVolume


def _make_vol(shape=(15, 15, 15), hkl_range=(-2, 2)):
    data = np.random.default_rng(42).uniform(0.5, 1.5, shape)
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    return HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]), ub_matrix=ub)


def test_bragg_mask_removes_integer_positions():
    vol = _make_vol()
    mask = bragg_mask(vol, punch_radius_hkl=0.4)
    # (0,0,0) is punched by the separate incident-beam path.
    ih0 = np.argmin(np.abs(vol.h_axis))
    ik0 = np.argmin(np.abs(vol.k_axis))
    il0 = np.argmin(np.abs(vol.l_axis))
    assert not mask[ih0, ik0, il0], "Incident beam (0,0,0) should be punched"


def test_bragg_mask_preserves_non_integer():
    vol = _make_vol()
    mask = bragg_mask(vol, punch_radius_hkl=0.25)
    # A voxel at hkl ≈ (0.5, 0.5, 0.5) should not be punched
    ih = np.argmin(np.abs(vol.h_axis - 0.5))
    ik = np.argmin(np.abs(vol.k_axis - 0.5))
    il = np.argmin(np.abs(vol.l_axis - 0.5))
    assert mask[ih, ik, il], "Non-integer HKL should not be punched"


def _peaky_vol(shape=(21, 21, 21), hkl_range=(-2, 2)):
    """Diffuse background with sharp Bragg peaks at SOME integer nodes only
    (mimicking systematic absences) and an off-integer peak centre."""
    rng = np.random.default_rng(7)
    data = rng.uniform(0.5, 1.5, shape)
    h = np.linspace(hkl_range[0], hkl_range[1], shape[0])
    k = np.linspace(hkl_range[0], hkl_range[1], shape[1])
    l = np.linspace(hkl_range[0], hkl_range[1], shape[2])
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(data, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1]),
                               ub_matrix=ub)
    present = {(0, 0, 0): 200.0, (1, 0, 0): 50.0, (-1, 1, 0): 30.0}
    for (h0, k0, l0), amp in present.items():
        ih = int(np.argmin(np.abs(vol.h_axis - h0)))
        ik = int(np.argmin(np.abs(vol.k_axis - k0)))
        il = int(np.argmin(np.abs(vol.l_axis - l0)))
        vol.data[ih, ik, il] = amp
    return vol, present


def test_bragg_detect_skips_absent_nodes():
    vol, present = _peaky_vol()
    remover = BraggRemover(punch_radii=(0.25, 0.25, 0.25), min_intensity=10.0)
    detected = remover.detect_peaks(vol)
    assert len(detected) == len(present) - 1      # real Bragg peaks; origin is separate
    # An empty node, e.g. (2,2,2), is NOT punched (preserve diffuse at absences).
    mask = remover.build_mask(vol)
    ih = int(np.argmin(np.abs(vol.h_axis - 2)))
    assert mask[ih, ih, ih]
    # ... while the incident beam IS punched by its separate mask.
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert not mask[i0, i0, i0]


def test_integer_shell_threshold_catches_weak_high_q_bragg():
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    ih = int(np.argmin(np.abs(vol.h_axis - 2)))
    ik = int(np.argmin(np.abs(vol.k_axis)))
    il = int(np.argmin(np.abs(vol.l_axis)))
    vol.data[ih, ik, il] = 2.4

    flat = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                        min_intensity=10.0, force_origin=False)
    shell = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                         min_intensity=None, integer_n_mad=2.0,
                         integer_q_step=0.4, min_prominence=0.5,
                         force_origin=False)

    assert flat.build_mask(vol)[ih, ik, il]
    assert not shell.build_mask(vol)[ih, ik, il]


def test_integer_shell_threshold_still_skips_extinct_nodes():
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    remover = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                           min_intensity=None, integer_n_mad=2.0,
                           integer_q_step=0.4, min_prominence=0.5,
                           force_origin=False)
    keep = remover.build_mask(vol)
    ih = int(np.argmin(np.abs(vol.h_axis - 2)))
    ik = int(np.argmin(np.abs(vol.k_axis - 2)))
    il = int(np.argmin(np.abs(vol.l_axis - 2)))
    assert keep[ih, ik, il]


def test_bragg_anisotropic_radii_punch_more_along_broad_axis():
    vol, _ = _peaky_vol()
    mask = bragg_mask(vol, punch_radii=(0.1, 0.1, 0.6), min_intensity=10.0)
    i0 = int(np.argmin(np.abs(vol.h_axis)))           # origin peak voxel
    punched = ~mask
    # Count punched voxels along H vs L lines through the origin: L (broad) > H.
    nh = int(punched[:, i0, i0].sum())
    nl = int(punched[i0, i0, :].sum())
    assert nl > nh


def test_bragg_intensity_scaling_enlarges_bright_peaks():
    vol, _ = _peaky_vol()
    ih = int(np.argmin(np.abs(vol.h_axis - 1)))
    i0 = int(np.argmin(np.abs(vol.k_axis)))
    base = BraggRemover(punch_radii=(0.2, 0.2, 0.2), min_intensity=10.0)
    scaled = BraggRemover(punch_radii=(0.2, 0.2, 0.2), min_intensity=10.0,
                          intensity_scale=True, intensity_ref=30.0)
    n_base = int((~base.build_mask(vol))[ih, :, i0].sum())
    n_scaled = int((~scaled.build_mask(vol))[ih, :, i0].sum())
    assert n_scaled > n_base                          # the bright Bragg peak grows


def test_integer_peak_fit_records_subvoxel_center_and_anisotropic_shape():
    shape = (51, 51, 51)
    h = np.linspace(-2.5, 2.5, shape[0])
    k = np.linspace(-2.5, 2.5, shape[1])
    l = np.linspace(-2.5, 2.5, shape[2])
    vol = HKLVolume.from_arrays(
        np.ones(shape) * 0.2, (h[0], h[-1]), (k[0], k[-1]), (l[0], l[-1])
    )
    H, K, L = vol.hkl_grid()
    # A Bragg peak near integer node (1,0,0), shifted off the grid and broader
    # along L.  The fitter should preserve the nearby integer-node decision but
    # punch the measured peak's fitted position/shape.
    vol.data += 30.0 * np.exp(-0.5 * (
        ((H - 1.07) / 0.05) ** 2
        + ((K - 0.02) / 0.05) ** 2
        + ((L + 0.03) / 0.22) ** 2
    ))

    remover = BraggRemover(
        mode="integer",
        punch_radii=(0.08, 0.08, 0.08),
        min_intensity=5.0,
        min_prominence=2.0,
        detect_window_hkl=0.35,
        integer_optimize_position=True,
        integer_optimize_shape=True,
        integer_fit_threshold_frac=0.2,
        integer_fit_radius_n_sigma=2.0,
        force_origin=False,
    )
    peak = remover._detect_peak_records(vol)[0]
    assert peak.source_node_hkl == (1, 0, 0)
    assert abs(peak.center_hkl[0] - 1.07) < 0.04
    assert peak.radii_hkl is not None
    assert peak.radii_hkl[2] > peak.radii_hkl[0]

    keep = remover.build_mask(vol)
    ih = int(np.argmin(np.abs(vol.h_axis - peak.center_hkl[0])))
    ik = int(np.argmin(np.abs(vol.k_axis - peak.center_hkl[1])))
    assert int((~keep)[ih, ik, :].sum()) > int((~keep)[:, ik, peak.il].sum())


def test_integer_h_guard_prevents_bleed_into_fractional_h_planes():
    data = np.ones((13, 13, 13), dtype=float) * 0.2
    vol = HKLVolume.from_arrays(data, (-1, 1), (-2, 2), (-2, 2))
    ih0 = int(np.argmin(np.abs(vol.h_axis)))
    ik1 = int(np.argmin(np.abs(vol.k_axis - 1)))
    il0 = int(np.argmin(np.abs(vol.l_axis)))
    ih_frac = int(np.argmin(np.abs(vol.h_axis - (1.0 / 3.0))))
    vol.data[ih0, ik1, il0] = 20.0

    wide = BraggRemover(
        mode="integer",
        punch_radii=(0.45, 0.12, 0.12),
        min_intensity=5.0,
        min_prominence=2.0,
        force_origin=False,
    )
    guarded = BraggRemover(
        mode="integer",
        punch_radii=(0.45, 0.12, 0.12),
        min_intensity=5.0,
        min_prominence=2.0,
        integer_h_guard_hkl=0.12,
        force_origin=False,
    )

    assert not wide.build_mask(vol)[ih_frac, ik1, il0]
    assert guarded.build_mask(vol)[ih_frac, ik1, il0]
    assert not guarded.build_mask(vol)[ih0, ik1, il0]


def test_q_shell_backfill_uses_radial_background_level():
    shape = (31, 31, 31)
    vol = HKLVolume.from_arrays(np.full(shape, 0.5), (-2, 2), (-2, 2), (-2, 2))
    q = vol.q_magnitude()
    ih = int(np.argmin(np.abs(vol.h_axis - 1.0)))
    ik = int(np.argmin(np.abs(vol.k_axis)))
    il = int(np.argmin(np.abs(vol.l_axis)))
    q0 = float(q[ih, ik, il])
    radial_band = np.abs(q - q0) <= 0.06
    vol.data[radial_band] = 2.0
    vol.data[ih, ik, il] = 100.0
    vol.mask[ih, ik, il] = False

    filled = backfill_bragg(
        vol, method="q_shell", q_shell_step=0.12, q_shell_min_count=8,
        direct_beam_fill=False,
    )

    assert filled.mask.all()
    assert abs(float(filled.data[ih, ik, il]) - 2.0) < 1e-12


def test_search_mode_punches_off_integer_satellite():
    """Search mode catches a sharp peak at a NON-integer position that the
    integer mode misses (e.g. a superlattice / small-domain satellite)."""
    vol, _ = _peaky_vol()
    # add a sharp satellite at a half-integer position
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0

    integer = BraggRemover(mode="integer", punch_radii=(0.25, 0.25, 0.25),
                           min_intensity=10.0)
    search = BraggRemover(mode="search", punch_radii=(0.25, 0.25, 0.25),
                          search_n_mad=6.0, search_min_intensity=10.0,
                          search_q_step=0.5)

    assert integer.build_mask(vol)[sh, sk, sl]        # integer mode leaves it
    assert not search.build_mask(vol)[sh, sk, sl]     # search mode punches it
    # search also removes an off-origin integer Bragg (a sharp outlier on a
    # well-populated |Q| shell); (0,0,0) at |Q|=0 is a sparse-shell edge case.
    i1h = int(np.argmin(np.abs(vol.h_axis - 1)))
    i1 = int(np.argmin(np.abs(vol.k_axis)))
    assert not search.build_mask(vol)[i1h, i1, i1]


def test_search_mode_punches_incident_beam_separately():
    vol, _ = _peaky_vol()
    search = BraggRemover(mode="search", punch_radii=(0.25, 0.25, 0.25),
                          search_n_mad=6.0, search_min_intensity=10.0,
                          search_q_step=0.5)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert all((p[0], p[1], p[2]) != (i0, i0, i0) for p in search.detect_peaks(vol))
    assert not search.build_mask(vol)[i0, i0, i0]


def test_incident_beam_sphere_punches_isotropic_origin_region():
    vol, _ = _peaky_vol(shape=(41, 41, 41), hkl_range=(-2, 2))
    remover = BraggRemover(
        mode="search",
        punch_radii=(0.1, 0.1, 0.1),
        search_min_intensity=1e6,
        incident_beam_sphere_radius_hkl=0.8,
    )
    keep = remover.build_mask(vol)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    i07 = int(np.argmin(np.abs(vol.h_axis - 0.7)))
    i10 = int(np.argmin(np.abs(vol.h_axis - 1.0)))

    assert not keep[i0, i0, i0]
    assert not keep[i07, i0, i0]
    assert keep[i10, i0, i0]


def test_incident_beam_ellipsoid_punches_anisotropic_origin_region():
    vol, _ = _peaky_vol(shape=(41, 41, 41), hkl_range=(-2, 2))
    # rh=0.3, rk=0.8, rl=1.5 — deliberately different so we can verify each axis
    remover = BraggRemover(
        mode="search",
        punch_radii=(0.1, 0.1, 0.1),
        search_min_intensity=1e6,
        incident_beam_ellipsoid_radii_hkl=(0.3, 0.8, 1.5),
    )
    keep = remover.build_mask(vol)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    # Inside ellipsoid in all three axes → punched
    assert not keep[i0, i0, i0]
    # H=0.25 < rh=0.3 → inside → punched
    ih025 = int(np.argmin(np.abs(vol.h_axis - 0.25)))
    assert not keep[ih025, i0, i0]
    # H=0.35 > rh=0.3 → outside → kept
    ih035 = int(np.argmin(np.abs(vol.h_axis - 0.35)))
    assert keep[ih035, i0, i0]
    # K=0.7 < rk=0.8 → inside → punched
    ik07 = int(np.argmin(np.abs(vol.k_axis - 0.7)))
    assert not keep[i0, ik07, i0]
    # K=0.9 > rk=0.8 → outside → kept
    ik09 = int(np.argmin(np.abs(vol.k_axis - 0.9)))
    assert keep[i0, ik09, i0]


def test_incident_beam_ellipsoid_takes_precedence_over_sphere():
    vol, _ = _peaky_vol(shape=(41, 41, 41), hkl_range=(-2, 2))
    # sphere r=0.1 would keep H=0.5; ellipsoid rh=0.6 would punch it
    remover = BraggRemover(
        mode="search",
        punch_radii=(0.1, 0.1, 0.1),
        search_min_intensity=1e6,
        incident_beam_ellipsoid_radii_hkl=(0.6, 0.6, 0.6),
        incident_beam_sphere_radius_hkl=0.1,
    )
    keep = remover.build_mask(vol)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    ih05 = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    assert not keep[ih05, i0, i0]


def test_phi_tail_expands_punch_along_ring_tangent():
    vol, _ = _peaky_vol()
    ih = int(np.argmin(np.abs(vol.h_axis - 0)))
    ik = int(np.argmin(np.abs(vol.k_axis - 1)))
    il = int(np.argmin(np.abs(vol.l_axis - 0)))
    vol.data[ih, ik, il] = 100.0

    base = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                        min_intensity=10.0, force_origin=False)
    phi = BraggRemover(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                       min_intensity=10.0, force_origin=False,
                       phi_tail_hkl=0.4)

    base_line = int((~base.build_mask(vol))[ih, ik, :].sum())
    phi_line = int((~phi.build_mask(vol))[ih, ik, :].sum())
    assert phi_line > base_line


def test_phi_tail_uses_ub_metric_ring_tangent():
    vol, _ = _peaky_vol(shape=(41, 41, 41), hkl_range=(-2, 2))
    vol.ub_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 3.0, 1.2],
        [0.0, 0.4, 0.7],
    ])
    direction = BraggRemover._kl_ring_directions(vol, (0.0, 1.0, 1.0))
    assert direction is not None
    krad, lrad, ktan, ltan = direction

    # The metric-aware tangent is perpendicular to grad(|Q|^2), not simply
    # (-L, K).  For this skewed UB those are measurably different directions.
    naive = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    actual = np.array([ktan, ltan])
    assert abs(float(np.dot(actual, naive))) < 0.95

    metric = vol.ub_matrix.T @ vol.ub_matrix
    grad_kl = np.array((metric @ np.array([0.0, 1.0, 1.0]))[1:3])
    assert abs(float(np.dot(grad_kl, actual))) < 1e-12

    # Finite-difference guard: moving along the returned tangent should not
    # change the physical |Q| to first order.
    eps = 1e-5
    q_plus = np.linalg.norm(
        np.array([0.0, 1.0 + eps * ktan, 1.0 + eps * ltan]) @ vol.ub_matrix.T
    )
    q_minus = np.linalg.norm(
        np.array([0.0, 1.0 - eps * ktan, 1.0 - eps * ltan]) @ vol.ub_matrix.T
    )
    assert abs((q_plus - q_minus) / (2.0 * eps)) < 1e-10


def test_auto_mode_aliases_search_mode():
    vol, _ = _peaky_vol()
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0
    auto = BraggRemover(mode="auto", punch_radii=(0.25, 0.25, 0.25),
                        search_n_mad=6.0, search_min_intensity=10.0,
                        search_q_step=0.5)
    assert not auto.build_mask(vol)[sh, sk, sl]


def test_search_prominence_rejects_broad_diffuse_bump():
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    H, K, L = vol.hkl_grid()
    broad = 5.0 * np.exp(-0.5 * (((H - 0.6) / 0.45) ** 2
                                  + ((K - 0.4) / 0.45) ** 2
                                  + ((L - 0.2) / 0.45) ** 2))
    vol.data += broad
    ih = int(np.argmin(np.abs(vol.h_axis - 0.6)))
    ik = int(np.argmin(np.abs(vol.k_axis - 0.4)))
    il = int(np.argmin(np.abs(vol.l_axis - 0.2)))

    sh = int(np.argmin(np.abs(vol.h_axis + 1.4)))
    sk = int(np.argmin(np.abs(vol.k_axis - 1.2)))
    sl = int(np.argmin(np.abs(vol.l_axis + 0.8)))
    vol.data[sh, sk, sl] = 8.0

    auto = BraggRemover(mode="auto", punch_radii=(0.2, 0.2, 0.2),
                        search_n_mad=3.0, search_min_intensity=1.0,
                        search_min_prominence=1.0, search_q_step=0.5)
    keep = auto.build_mask(vol)
    assert keep[ih, ik, il]           # broad diffuse maximum survives
    assert not keep[sh, sk, sl]       # sharp satellite is punched


def test_search_exclude_h_protects_fractional_diffuse_plane():
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    protected_h = int(np.argmin(np.abs(vol.h_axis - (1.0 / 3.0))))
    protected_k = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    protected_l = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    free_h = int(np.argmin(np.abs(vol.h_axis - 1.5)))
    free_k = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    free_l = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[protected_h, protected_k, protected_l] = 60.0
    vol.data[free_h, free_k, free_l] = 60.0

    remover = BraggRemover(
        mode="search",
        punch_radii=(0.25, 0.25, 0.25),
        search_n_mad=4.0,
        search_min_intensity=10.0,
        search_q_step=0.5,
        search_exclude_h_centers=(1.0 / 3.0,),
        search_exclude_h_half_width=0.08,
        force_origin=False,
    )
    keep = remover.build_mask(vol)
    assert keep[protected_h, protected_k, protected_l]
    assert not keep[free_h, free_k, free_l]


def test_integer_local_prominence_catches_small_sharp_bragg():
    """A small but sharp peak at an integer node, below the absolute intensity
    floor, is caught only when the local relative-prominence criterion is on."""
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    ih = int(np.argmin(np.abs(vol.h_axis - 2)))
    ik = int(np.argmin(np.abs(vol.k_axis)))
    il = int(np.argmin(np.abs(vol.l_axis)))
    vol.data[ih, ik, il] = 4.0   # sharp, but well below an absolute floor of 10

    common = dict(mode="integer", punch_radii=(0.2, 0.2, 0.2),
                  min_intensity=10.0, force_origin=False)
    # absolute floor alone misses the small peak ...
    assert BraggRemover(**common).build_mask(vol)[ih, ik, il]
    # ... the local relative-prominence catch removes it.
    sharp = BraggRemover(**common, integer_local_prominence_n_mad=4.0)
    assert not sharp.build_mask(vol)[ih, ik, il]


def test_search_exclude_h_fractions_protects_thirds_family():
    """Periodic fractional protection shields every integer±1/3 plane (e.g.
    H=4/3), which a fixed (±1/3, ±2/3) centre list would not cover."""
    vol, _ = _peaky_vol(shape=(31, 31, 31), hkl_range=(-3, 3))
    prot_h = int(np.argmin(np.abs(vol.h_axis - 4.0 / 3.0)))   # grid 1.4, frac≈1/3
    free_h = int(np.argmin(np.abs(vol.h_axis - 1.8)))          # frac 0.8, not a third
    ik = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    il = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    assert prot_h != free_h
    vol.data[prot_h, ik, il] = 60.0
    vol.data[free_h, ik, il] = 60.0

    remover = BraggRemover(
        mode="search", punch_radii=(0.25, 0.25, 0.25),
        search_n_mad=4.0, search_min_intensity=10.0, search_q_step=0.5,
        search_exclude_h_fractions=(1.0 / 3.0, 2.0 / 3.0),
        search_exclude_h_half_width=0.08, force_origin=False,
    )
    keep = remover.build_mask(vol)
    assert keep[prot_h, ik, il]        # H=4/3 thirds-family plane protected
    assert not keep[free_h, ik, il]    # H=1.8 punched


def test_both_mode_is_sequential_union():
    """'both' = integer punch, then search on the residual; it removes both an
    integer peak and an off-integer satellite."""
    vol, _ = _peaky_vol()
    sh = int(np.argmin(np.abs(vol.h_axis - 0.5)))
    sk = int(np.argmin(np.abs(vol.k_axis - 0.5)))
    sl = int(np.argmin(np.abs(vol.l_axis - 1.0)))
    vol.data[sh, sk, sl] = 60.0
    both = BraggRemover(mode="both", punch_radii=(0.25, 0.25, 0.25),
                        min_intensity=10.0, search_n_mad=6.0,
                        search_min_intensity=10.0, search_q_step=0.5)
    keep = both.build_mask(vol)
    i0 = int(np.argmin(np.abs(vol.h_axis)))
    assert not keep[i0, i0, i0]      # integer Bragg gone
    assert not keep[sh, sk, sl]      # satellite gone


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(7) == 8
    assert _next_power_of_2(16) == 16
    assert _next_power_of_2(17) == 32


def test_delta_pdf_shape_and_finite():
    vol = _make_vol(shape=(8, 8, 8))
    dpdf = compute_delta_pdf(vol, apodization="hann", zero_pad=False)
    assert dpdf.data.shape == (8, 8, 8)
    assert np.isfinite(dpdf.data).all()


def test_delta_pdf_zero_pad_increases_size():
    vol = _make_vol(shape=(10, 10, 10))
    dpdf = compute_delta_pdf(vol, apodization="none", zero_pad=True)
    assert all(s >= 10 for s in dpdf.data.shape)
    # next power of 2 after 10 is 16
    assert dpdf.data.shape == (16, 16, 16)


def test_delta_pdf_hk0_slice():
    vol = _make_vol(shape=(8, 8, 8))
    dpdf = compute_delta_pdf(vol, apodization="none", zero_pad=False)
    sl = dpdf.slice_hk0()
    assert sl.shape == (8, 8)


def test_delta_pdf_centring_positive_peak():
    """A single positive cosine correlation must give POSITIVE real-space peaks.

    Regression guard for the FFT origin-centring bug: the input has Q=0 at the
    array centre, so the transform needs ifftshift before fftn.  Without it the
    output picks up a (-1)^k phase ramp that flips peak signs by pixel parity
    (each correlation splits into mixed +/- lobes).  With I(Q)=1+cos(2π·3·(i-c)/N)
    — even about the Q=0 centre — the buggy transform produced -2048 where the
    correct one produces +2048.
    """
    N, c = 16, 8
    idx = np.arange(N)
    line = 1.0 + np.cos(2 * np.pi * 3 * (idx - c) / N)  # even about Q=0
    data = np.broadcast_to(line[:, None, None], (N, N, N)).astype(float).copy()
    ub = 2 * np.pi * np.eye(3) / 4.0
    vol = HKLVolume.from_arrays(data, (-2, 2), (-2, 2), (-2, 2), ub_matrix=ub)

    dpdf = compute_delta_pdf(
        vol, apodization="none", zero_pad=False,
        subtract_mean=True, real_space_angstrom=False,
    )
    hline = dpdf.data[:, c, c]
    peaks = np.where(np.abs(hline) > 0.5 * np.abs(hline).max())[0]

    assert list(peaks) == [c - 3, c + 3]           # peaks at the right distance
    assert (hline[peaks] > 0).all()                # correct sign (was negative)
    assert np.isclose(hline[c - 3], hline[c + 3])  # centrosymmetric
