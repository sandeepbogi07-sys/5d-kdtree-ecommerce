import sys, os, random, math, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kdtree import KDTree, brute_knn, brute_radius
from distance import feature_vector, distance, norm
from app import create_app

def pts(n, seed=1):
    r = random.Random(seed); return [(i, [r.random() for _ in range(5)]) for i in range(n)]
def ids(res): return [i for i, _ in res]

@pytest.mark.parametrize("w", [None, [1]*5, [0, 3, 0.5, 2, 0], [0]*5])
def test_knn_matches_brute(w):
    items = pts(300); t = KDTree(); t.build(items); ww = w or [1]*5
    for q in [p[1] for p in pts(15, 9)]:
        for k in (1, 5, 400):
            a, _ = t.knn(q, k, w); b, _ = brute_knn(items, q, k, ww)
            assert ids(a) == ids(b)
def test_radius_matches_brute():
    items = pts(300); t = KDTree(); t.build(items)
    for q in [p[1] for p in pts(10, 5)]:
        assert ids(t.radius(q, 0.8)[0]) == ids(brute_radius(items, q, 0.8, [1]*5)[0])
def test_pruning_happens():
    items = pts(2000); t = KDTree(); t.build(items); assert t.knn(items[0][1], 3)[1]["dist_calcs"] < 2000
def test_empty_single_and_k_big():
    t = KDTree(); t.build([]); assert t.knn([0]*5, 3)[0] == [] and t.stats()["height"] == 0
    t.build([(1, [.1]*5)]); assert ids(t.knn([0]*5, 10)[0]) == [1]
def test_duplicates_ties_by_id():
    t = KDTree(); t.build([(i, [.5]*5) for i in (7, 3, 9, 1)])
    assert ids(t.knn([.5]*5, 3)[0]) == [1, 3, 7]
def test_invalid_inputs():
    t = KDTree(); t.build(pts(5))
    for bad in (-1, float("nan")):
        with pytest.raises(ValueError): t.radius([0]*5, bad)
    with pytest.raises(ValueError): t.knn([0]*5, 0)
    with pytest.raises(ValueError): KDTree([1, -1, 1, 1, 1])
    with pytest.raises(ValueError): KDTree([1, 1, 1])
def test_weighted_distance():
    assert distance([0]*5, [1, 1, 0, 0, 0], [4, 9, 1, 1, 1]) == pytest.approx(math.sqrt(13))
def test_circular_hue():
    a, b, c = (feature_vector(1000, 200, 10, h) for h in (359, 1, 180))
    assert distance(a, b) < 0.1 < distance(a, c)
def test_zero_range_and_missing():
    assert norm(5, 3, 3) == 0.0
    with pytest.raises(ValueError): norm(None, 0, 1)
def test_insert_delete_rebuild():
    items = pts(50); t = KDTree(); t.build(items); t.insert(100, [0.5]*5)
    assert ids(t.knn([0.5]*5, 1)[0]) == [100]
    assert t.delete(100) and not t.delete(100) and 100 not in ids(t.knn([.5]*5, 5)[0])
    assert t.stats()["deleted_nodes"] == 1; t.rebuild(); s = t.stats(); assert s["deleted_nodes"] == 0 and s["active_nodes"] == 50

@pytest.fixture
def client(tmp_path):
    return create_app().test_client()
H = {"X-Admin-Token": "dev-admin-token"}
P = dict(name="A", category="Shirt", price=999, gsm=150, elasticity=5, hue=200)
def test_api_flow(client):
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/products", json=P).status_code == 401
    assert client.post("/api/products", json={**P, "price": -1}, headers=H).status_code == 422
    r = client.post("/api/products", json=P, headers=H); assert r.status_code == 201; a = r.get_json()["id"]
    b = client.post("/api/products", json={**P, "name": "B", "hue": 205}, headers=H).get_json()["id"]
    rec = client.get(f"/api/recommendations/{a}?k=3&compare=1").get_json()
    assert b in [x["product"]["id"] for x in rec["results"]] and rec["brute_force"]["agrees"]
    assert client.get("/api/recommendations/999").status_code == 404
    assert client.post("/api/search/radius", json={"product_id": a, "radius": -1}).status_code == 422
    assert client.post("/api/search/knn", json={"product_id": a, "weights": [-1,1,1,1,1]}).status_code == 422
    assert client.delete(f"/api/products/{b}", headers=H).status_code == 200
    assert b not in [x["product"]["id"] for x in client.get(f"/api/recommendations/{a}").get_json()["results"]]
    st = client.get("/api/kdtree/stats").get_json(); assert st["deleted_nodes"] == 1
    assert client.post("/api/kdtree/rebuild", headers=H).get_json()["deleted_nodes"] == 0
def test_visualization_endpoint(client):
    data = client.get("/api/kdtree/visualization").get_json()
    assert data["total_nodes"] == 160
    assert 0 < data["shown"] <= 31
    assert data["nodes"][0]["axis_name"] == "price"


def test_api_rejects_bad_vectors_and_fields(client):
    assert client.post("/api/search/knn", json={"vector":[0,0,float("nan"),0,0]}).status_code == 422
    assert client.post("/api/search/knn", json={"vector":[0,0,0,0,0],"k":9999}).status_code == 422
    assert client.post("/api/products", json={**P,"rating":9}, headers=H).status_code == 422
    assert client.post("/api/products", json={**P,"stock":1.5}, headers=H).status_code == 422
# ===================== EDGE-CASE TESTS (Member 1) =====================
# Paste everything from here to the end of backend/tests/test_all.py
import math, os, random, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from distance import feature_vector, norm, check_weights, distance
from kdtree import KDTree, brute_knn, brute_radius


def make_tree(points):
    """Helper: build a KDTree from a list of (pid, vector)."""
    t = KDTree()
    t.build(points)
    return t


def random_points(n, seed=1):
    rng = random.Random(seed)
    return [(i, [rng.random() for _ in range(5)]) for i in range(1, n + 1)]


# ---------- empty / tiny trees ----------
def test_knn_on_empty_tree_returns_nothing():
    res, st = KDTree().knn([0.5] * 5, 3)
    assert res == [] and st["visited"] == 0


def test_radius_on_empty_tree_returns_nothing():
    res, _ = KDTree().radius([0.5] * 5, 1.0)
    assert res == []


def test_single_point_tree():
    t = make_tree([(7, [0.1, 0.2, 0.3, 1.0, 0.0])])
    res, _ = t.knn([0.1, 0.2, 0.3, 1.0, 0.0], 1)
    assert res == [(7, 0.0)]


def test_k_larger_than_n_returns_all_points():
    pts = random_points(4)
    res, _ = make_tree(pts).knn([0.5] * 5, 50)
    assert len(res) == 4


# ---------- invalid inputs ----------
@pytest.mark.parametrize("bad_k", [0, -1, 2.5, True, "3", None])
def test_knn_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        make_tree(random_points(5)).knn([0.5] * 5, bad_k)


@pytest.mark.parametrize("bad_r", [-0.1, float("nan"), float("inf"), True, "1"])
def test_radius_rejects_invalid_radius(bad_r):
    with pytest.raises(ValueError):
        make_tree(random_points(5)).radius([0.5] * 5, bad_r)


@pytest.mark.parametrize("bad_w", [[1, 1, 1], [1, 1, 1, 1, -1], [1, 1, 1, 1, float("nan")], [1, 1, 1, 1, "a"]])
def test_check_weights_rejects_bad_weights(bad_w):
    with pytest.raises(ValueError):
        check_weights(bad_w)


def test_check_weights_default_is_all_ones():
    assert check_weights(None) == [1.0] * 5


# ---------- radius behaviour ----------
def test_radius_zero_returns_only_exact_match():
    pts = random_points(20)
    res, _ = make_tree(pts).radius(pts[3][1], 0)
    assert [pid for pid, _ in res] == [pts[3][0]]


def test_radius_huge_returns_everything():
    pts = random_points(20)
    res, _ = make_tree(pts).radius([0.5] * 5, 100)
    assert len(res) == 20


# ---------- ties and duplicates ----------
def test_identical_vectors_are_ordered_by_id():
    v = [0.5, 0.5, 0.5, 1.0, 0.0]
    t = make_tree([(3, v), (1, v), (2, v)])
    res, _ = t.knn(v, 3)
    assert [pid for pid, _ in res] == [1, 2, 3]


def test_insert_duplicate_id_raises():
    t = make_tree(random_points(5))
    with pytest.raises(ValueError):
        t.insert(1, [0.1] * 5)


def test_insert_into_empty_tree_becomes_root():
    t = KDTree()
    t.insert(1, [0.2] * 5)
    assert t.root.pid == 1 and t.find(1) is not None


def test_insert_then_found_by_knn():
    t = make_tree(random_points(30))
    t.insert(999, [0.5] * 5)
    res, _ = t.knn([0.5] * 5, 1)
    assert res[0][0] == 999


# ---------- lazy deletion ----------
def test_deleted_point_not_returned_by_knn():
    pts = random_points(30)
    t = make_tree(pts)
    target = pts[5]
    assert t.delete(target[0]) is True
    res, _ = t.knn(target[1], 30)
    assert target[0] not in [pid for pid, _ in res]


def test_delete_missing_id_returns_false():
    assert make_tree(random_points(5)).delete(12345) is False


def test_delete_same_id_twice_returns_false_second_time():
    t = make_tree(random_points(5))
    assert t.delete(1) is True
    assert t.delete(1) is False


def test_stats_and_rebuild_after_delete():
    t = make_tree(random_points(10))
    t.delete(1); t.delete(2)
    s = t.stats()
    assert s["active_nodes"] == 8 and s["deleted_nodes"] == 2
    t.rebuild()
    s = t.stats()
    assert s["active_nodes"] == 8 and s["deleted_nodes"] == 0


def test_delete_everything_then_search_is_empty():
    t = make_tree(random_points(5))
    for i in range(1, 6):
        t.delete(i)
    assert t.knn([0.5] * 5, 3)[0] == []


def test_reinsert_after_delete_works():
    t = make_tree(random_points(10))
    t.delete(4)
    t.insert(4, [0.9] * 5)
    assert t.find(4) is not None


# ---------- weights ----------
def test_zero_weight_axis_is_ignored_in_distance():
    # Two points differ only on axis 0; with weight 0 on axis 0 they are at distance 0.
    t = make_tree([(1, [0.0, 0.5, 0.5, 1.0, 0.0]), (2, [1.0, 0.5, 0.5, 1.0, 0.0])])
    res, _ = t.knn([0.0, 0.5, 0.5, 1.0, 0.0], 2, [0, 1, 1, 1, 1])
    assert [d for _, d in res] == [0.0, 0.0]


# ---------- feature vector / hue ----------
def test_hue_360_equals_hue_0():
    a = feature_vector(1000, 200, 10, 0)
    b = feature_vector(1000, 200, 10, 360)
    assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))


def test_hue_wraparound_359_is_close_to_1():
    a = feature_vector(1000, 200, 10, 359)
    b = feature_vector(1000, 200, 10, 1)
    c = feature_vector(1000, 200, 10, 180)
    assert distance(a, b) < 0.1 < distance(a, c)


def test_negative_hue_wraps_correctly():
    a = feature_vector(1000, 200, 10, -30)
    b = feature_vector(1000, 200, 10, 330)
    assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))


def test_values_outside_range_are_clipped():
    v = feature_vector(10**9, 10**6, 10**6, 0)
    assert v[:3] == [1.0, 1.0, 1.0]
    v = feature_vector(-5, -5, -5, 0)
    assert v[:3] == [0.0, 0.0, 0.0]


def test_nan_and_none_values_rejected():
    with pytest.raises(ValueError):
        feature_vector(float("nan"), 200, 10, 0)
    with pytest.raises(ValueError):
        feature_vector(1000, 200, 10, None)


def test_norm_zero_range_returns_zero():
    assert norm(5, 3, 3) == 0.0


# ---------- KD-tree must agree with brute force ----------
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_knn_matches_brute_force_random(seed):
    pts = random_points(150, seed)
    t = make_tree(pts)
    q = [random.Random(seed + 100).random() for _ in range(5)]
    assert t.knn(q, 7)[0] == brute_knn(pts, q, 7, [1] * 5)[0]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_radius_matches_brute_force_random(seed):
    pts = random_points(150, seed)
    t = make_tree(pts)
    q = [0.5] * 5
    assert t.radius(q, 0.6)[0] == brute_radius(pts, q, 0.6, [1] * 5)[0]
