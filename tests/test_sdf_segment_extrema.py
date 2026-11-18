"""Reference interpolation tests, not GPU/physical-contact qualification."""
import importlib
import numpy as np
import pytest

NAME = "go2_pvcnn.control.dh116.tabletop_sdf_segment_extrema"

def module():
    assert importlib.util.find_spec(NAME) is not None, "continuous SDF segment audit is missing"
    return importlib.import_module(NAME)

def grid(fn, shape=(2, 2, 2), spacing=1.):
    ijk = np.indices(shape).transpose(1, 2, 3, 0) * spacing
    return dict(values=fn(*np.moveaxis(ijk, -1, 0)), origin=np.zeros(3), spacing=spacing)

def audit(g, a, b, radius=0.):
    return module().segment_extrema(g, a, b, radius=radius)

def test_interior_minimum_missed_by_endpoint_sampling():
    r = audit(grid(lambda x,y,z: (x-.5)*(y-.5)+0*z),
              [.1,.1,.5], [.9,.9,.5])
    assert r["minimum_gap_m"] == pytest.approx(0., abs=1e-13)
    assert r["minimum_at_fraction"] == pytest.approx(.5)
    assert len(r["segments"]) == 1
    assert r["gpu_bitwise_verified"] is False
    assert r["native_contact_verified"] is False

def test_cubic_interior_stationary_roots_are_considered():
    r = audit(grid(lambda x,y,z: x*y*z-.6*x*y+.09*x+0*z),
              [0,0,0], [1,1,1])
    roots = np.roots([3., -1.2, .09])
    points = [0, 1, *roots]
    expected = min(t**3-.6*t*t+.09*t for t in points)
    assert r["minimum_gap_m"] == pytest.approx(expected, abs=1e-13)
    assert sum(len(s["stationary_fractions"]) for s in r["segments"]) == 2

def test_cell_crossing_keeps_continuous_v_minimum():
    g = dict(values=np.array([1.,-1.,1.])[:,None,None]*np.ones((3,2,2)),
             origin=np.zeros(3), spacing=1.)
    r = audit(g,[.25,.5,.5],[1.75,.5,.5],.2)
    assert r["minimum_gap_m"] == pytest.approx(-1.2)
    assert r["minimum_at_fraction"] == pytest.approx(.5)
    assert len(r["segments"]) == 2

def test_reverse_line_preserves_extremum_and_reverses_fraction():
    g = grid(lambda x,y,z: x+2*y-3*z)
    a,b = [.1,.2,.3],[.8,.4,.5]
    r,s = audit(g,a,b),audit(g,b,a)
    assert r["minimum_gap_m"] == pytest.approx(s["minimum_gap_m"])
    assert r["minimum_at_fraction"] == pytest.approx(1-s["minimum_at_fraction"])

def test_partial_domain_reports_excluded_time_not_fake_zero_depth():
    r = audit(grid(lambda x,y,z: x+0*y+0*z),[-1,.5,.5],[2,.5,.5])
    assert r["domain_fraction"] == pytest.approx(1/3)
    assert r["minimum_gap_m"] == pytest.approx(0)
    assert r["outside_domain_semantics"] == "solver_sdf_sample_returns_false_not_physical_clearance"

def test_fixed_upper_face_is_outside_half_open_solver_lattice():
    r = audit(grid(lambda x,y,z: x+y+z),[1,.2,.2],[1,.7,.7])
    assert r["minimum_gap_m"] is None
    assert r["segments"] == []
    assert r["domain_fraction"] == 0

def test_stationary_inside_point_retains_original_distance():
    r = audit(grid(lambda x,y,z: x+y+z),[.25,.25,.25],[.25,.25,.25],.1)
    assert r["minimum_gap_m"] == pytest.approx(.65)
    assert r["domain_fraction"] == 1
    assert len(r["segments"]) == 1

def test_constant_field_has_no_false_stationary_root():
    r = audit(grid(lambda x,y,z: 4+0*x+0*y+0*z),[0,0,0],[1,1,1],.5)
    assert r["minimum_gap_m"] == pytest.approx(3.5)
    assert r["segments"][0]["stationary_fractions"] == []

def test_corner_touch_lower_boundary_is_not_silently_omitted():
    r = audit(grid(lambda x,y,z: x+y+z),[-1,-1,-1],[0,0,0])
    assert r["minimum_gap_m"] == pytest.approx(0)
    assert r["minimum_at_fraction"] == 1
    assert r["domain_fraction"] == 0

@pytest.mark.parametrize("bad", ["nan", "negative_spacing", "short_axis", "nan_point", "negative_radius"])
def test_invalid_contract_rejected(bad):
    g=grid(lambda x,y,z: x+y+z);a=np.array([.1,.1,.1]);radius=0.
    if bad=="nan":g["values"][0,0,0]=np.nan
    elif bad=="negative_spacing":g["spacing"]=-1
    elif bad=="short_axis":g["values"]=g["values"][:1]
    elif bad=="nan_point":a[1]=np.nan
    else:radius=-.01
    with pytest.raises(ValueError):audit(g,a,[.9,.9,.9],radius)

def test_multiple_cells_against_independent_dense_trilinear_interpolator():
    from scipy.interpolate import RegularGridInterpolator
    rng=np.random.default_rng(37)
    values=rng.normal(size=(5,6,4))
    g=dict(values=values,origin=np.array([-.2,.4,1.]),spacing=.007)
    a=np.array([.1,.3,.2]);b=np.array([3.9,4.9,2.9])
    r=audit(g,g["origin"]+a*.007,g["origin"]+b*.007,.005)
    t=np.linspace(0,1,20001)
    f=RegularGridInterpolator(tuple(np.arange(n) for n in values.shape),values)
    dense=f(a+(b-a)*t[:,None])-.005
    assert r["minimum_gap_m"] <= dense.min()+1e-12
    assert dense.min()-r["minimum_gap_m"] < 1e-3
    for s in r["segments"]:
        u=.317;fraction=s["start_fraction"]+u*(s["end_fraction"]-s["start_fraction"])
        prediction=np.polynomial.polynomial.polyval(u,s["phi_coefficients_ascending"])
        assert prediction == pytest.approx(float(f((a+(b-a)*fraction)[None])[0]),abs=1e-11)
