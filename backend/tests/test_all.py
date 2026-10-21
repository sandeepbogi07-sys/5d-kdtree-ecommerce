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
    # ---------- Edge-case tests (Member 1) ----------
def test_edge_knn_on_empty_tree_returns_empty():
    t = KDTree(); t.build([])
    res, st = t.knn([0.5] * 5, 5)
    assert res == [] and st["dist_calcs"] == 0

def test_edge_radius_zero_finds_exact_match_only():
    t = KDTree(); t.build([(1, [.5] * 5), (2, [.6] * 5)])
    assert ids(t.radius([.5] * 5, 0)[0]) == [1]

def test_edge_knn_rejects_bad_k_values():
    t = KDTree(); t.build(pts(5))
    for bad_k in (0, -3, 1.5, True):
        with pytest.raises(ValueError):
            t.knn([0] * 5, bad_k)

def test_edge_radius_rejects_infinite_radius():
    t = KDTree(); t.build(pts(5))
    with pytest.raises(ValueError):
        t.radius([0] * 5, float("inf"))
def test_edge_insert_duplicate_id_raises():
    t = KDTree(); t.build(pts(5))
    with pytest.raises(ValueError):
        t.insert(0, [.1] * 5)

def test_edge_insert_into_empty_tree():
    t = KDTree(); t.build([])
    t.insert(42, [.3] * 5)
    assert ids(t.knn([.3] * 5, 1)[0]) == [42] and t.stats()["height"] == 1

def test_edge_deleted_node_is_never_returned():
    items = pts(100); t = KDTree(); t.build(items)
    nearest = ids(t.knn(items[0][1], 1)[0])[0]
    assert t.delete(nearest)
    assert nearest not in ids(t.knn(items[0][1], 100)[0])
    assert nearest not in ids(t.radius(items[0][1], 10)[0])

def test_edge_delete_unknown_id_returns_false():
    t = KDTree(); t.build(pts(5))
    assert t.delete(99999) is False

def test_edge_reinsert_after_delete_works():
    t = KDTree(); t.build(pts(10)); t.delete(3)
    t.insert(3, [.9] * 5)
    assert 3 in ids(t.knn([.9] * 5, 1)[0])
def test_edge_api_missing_json_body_is_rejected(client):
    r = client.post("/api/search/knn", data="not json", content_type="text/plain")
    assert r.status_code in (400, 415, 422)

def test_edge_api_wrong_vector_length_is_rejected(client):
    assert client.post("/api/search/knn", json={"vector": [0.1, 0.2]}).status_code == 422

def test_edge_api_unknown_product_returns_404(client):
    assert client.get("/api/products/999999").status_code == 404
    assert client.delete("/api/products/999999", headers=H).status_code == 404

def test_edge_api_write_with_wrong_admin_token_is_blocked(client):
    r = client.post("/api/products", json=P, headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401

def test_edge_api_duplicate_product_name_is_rejected(client):
    assert client.post("/api/products", json={**P, "name": "Dup"}, headers=H).status_code == 201
    r = client.post("/api/products", json={**P, "name": "Dup"}, headers=H)
    assert r.status_code in (409, 422)

def test_edge_api_price_out_of_range_is_rejected(client):
    assert client.post("/api/products", json={**P, "price": 10**9}, headers=H).status_code == 422
