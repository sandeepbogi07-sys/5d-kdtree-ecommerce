"""Threadly - Flask API + static frontend for the 5D KD-tree clothing store (in-memory, no database).

FILE OVERVIEW (app.py)
----------------------
This file is the web server. It does three jobs:
    1. Serves the frontend page (frontend/index.html) at "/".
    2. Creates 160 sample clothing products in memory when the server starts.
    3. Exposes a JSON API (/api/...) that the page calls for products, KD-tree
       recommendations, searches, tree statistics, visualization and benchmarks.

Each product becomes a 5D vector (see distance.py):
    [normalised price, normalised GSM, normalised stretch, cos(hue), sin(hue)]
and is stored in the KD-tree (see kdtree.py) so nearest-neighbour search is fast.

Run from the project root:   python backend/app.py     then open http://127.0.0.1:5000
"""
import hmac, math, os, random, time          # hmac: safe token compare; random: sample data; time: timing metrics
from flask import Flask, jsonify, request, send_from_directory   # Flask web framework pieces
from flask_cors import CORS                  # allows the browser to call the API from other origins

from distance import feature_vector, check_weights   # product -> 5D vector; weight validation
from kdtree import KDTree, brute_knn, brute_radius   # our own KD-tree + slow brute-force baselines

# Names of the 5 dimensions, in vector order. Used by the tree visualization endpoint.
AXIS_NAMES = ["price", "gsm", "stretch", "hue_cos", "hue_sin"]
# Folder that holds index.html (one level up from backend/, then "frontend").
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
# Largest number of neighbours a request may ask for (protects the server from huge requests).
MAX_K = 500

# category -> (price range in rupees, GSM range, stretch % range). Used to generate realistic sample data.
CATEGORIES = {
    "T-Shirt":  ((399, 1799), (120, 220), (3, 12)),
    "Shirt":    ((799, 2999), (90, 180), (2, 8)),
    "Jeans":    ((1299, 4499), (280, 450), (0, 6)),
    "Trousers": ((999, 3999), (180, 320), (2, 10)),
    "Jacket":   ((2499, 7999), (300, 560), (0, 5)),
    "Dress":    ((899, 5999), (80, 240), (4, 40)),
}
# colour name -> base hue angle in degrees (0-360). Each product's hue is this value plus a small random shift.
COLORS = {"Red": 0, "Orange": 28, "Mustard": 48, "Olive": 75, "Green": 125, "Teal": 175,
          "Sky": 200, "Blue": 220, "Navy": 240, "Purple": 275, "Pink": 330}
# Words used to build product names such as "Urban Navy Jacket".
STYLES = ["Classic", "Urban", "Essential", "Premium", "Relaxed", "Slim", "Everyday", "Heritage"]


class ValidationError(ValueError):
    """Raised when user input is invalid. The API turns it into an HTTP 422 response."""
    pass


def _num(v, name, lo=None, hi=None, integer=False):
    """Validate one numeric field.
    Rejects: booleans, non-numbers, NaN/infinity, non-whole numbers (if integer=True),
    and values outside [lo, hi]. Returns the cleaned number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValidationError(f"{name} must be a finite number")
    if integer and (isinstance(v, float) and not v.is_integer()):
        raise ValidationError(f"{name} must be a whole number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValidationError(f"{name} must be between {lo} and {hi}")
    return int(v) if integer else v


def _color_for_hue(h):
    """Pick the closest named colour for a hue angle. Distance on a circle: 359 and 1 degree are close."""
    h %= 360
    return min(COLORS, key=lambda c: min(abs(COLORS[c] - h), 360 - abs(COLORS[c] - h)))


def clean_product(data, partial_base=None):
    """Validate a product payload and return a clean product dict.
    If partial_base is given (used by PUT/update), missing fields fall back to the existing product."""
    if not isinstance(data, dict):
        raise ValidationError("JSON object expected")
    d = dict(partial_base or {}); d.update(data)     # merge old values with the new ones
    # These fields are mandatory.
    for req in ("name", "category", "price", "gsm", "elasticity", "hue"):
        if req not in d: raise ValidationError(f"{req} is required")
    name = d["name"]
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValidationError("name must be 1-80 characters")
    if d["category"] not in CATEGORIES:
        raise ValidationError("category must be one of: " + ", ".join(CATEGORIES))
    p = {
        "name": name.strip(), "category": d["category"],
        "price": _num(d["price"], "price", 1, 100000),
        "gsm": _num(d["gsm"], "gsm", 1, 2000),
        "elasticity": _num(d["elasticity"], "elasticity", 0, 100),
        "hue": _num(d["hue"], "hue", -3600, 3600) % 360,          # wrap hue into 0-360
        "rating": _num(d.get("rating", 4.0), "rating", 0, 5),     # optional, default 4.0
        "stock": _num(d.get("stock", 10), "stock", 0, 100000, integer=True),   # optional, whole number
    }
    # Colour name: use the given one, otherwise derive it from the hue.
    c = d.get("color")
    p["color"] = c if isinstance(c, str) and c.strip() else _color_for_hue(p["hue"])
    # Description: use the given one, otherwise generate a simple sentence.
    desc = d.get("description")
    p["description"] = desc if isinstance(desc, str) else f"{p['name']} - a {p['category'].lower()} in {p['color'].lower()}."
    img = d.get("image")                                          # optional image URL
    if isinstance(img, str): p["image"] = img
    return p


def vec_of(p):
    """Convert a product dict into its 5D feature vector (normalised price, GSM, stretch, cos hue, sin hue)."""
    return feature_vector(p["price"], p["gsm"], p["elasticity"], p["hue"])


def generate_products(n=160, seed=42):
    """Create n sample products. A fixed seed makes the data identical on every start (deterministic)."""
    r = random.Random(seed); out = []; used = set()     # `used` prevents duplicate product names
    cats = list(CATEGORIES)
    for i in range(1, n + 1):
        cat = cats[(i - 1) % len(cats)]                  # rotate through categories evenly
        (plo, phi), (glo, ghi), (elo, ehi) = CATEGORIES[cat]
        color = r.choice(list(COLORS)); hue = (COLORS[color] + r.uniform(-6, 6)) % 360
        while True:                                       # keep trying until the name is unique
            nm = f"{r.choice(STYLES)} {color} {cat}"
            if nm in used: nm += f" {r.randint(2, 99)}"
            if nm not in used: used.add(nm); break
        out.append({
            "id": i, "name": nm, "category": cat, "color": color, "hue": round(hue, 1),
            "price": int(round(r.uniform(plo, phi) / 10) * 10),     # rounded to nearest 10 rupees
            "gsm": r.randint(glo, ghi), "elasticity": round(r.uniform(elo, ehi), 1),
            "rating": round(r.uniform(3.2, 5.0), 1),
            "stock": r.choice([0, 3, 5, 8, 12, 20, 35, 50]),         # includes 0 (sold out) and low stock
            "description": f"A {color.lower()} {cat.lower()} with a comfortable {glo}-{ghi} GSM fabric feel.",
        })
    return out


def create_app():
    """Application factory: builds a fresh Flask app with its own products and KD-tree.
    The tests call this to get a clean app each time."""
    app = Flask(__name__, static_folder=None)
    CORS(app)
    # S = all server state, kept in memory (no database):
    #   products - dict: product id -> product dict
    #   next_id  - the id given to the next new product (ids are never reused)
    #   tree     - the KD-tree holding every product's 5D vector
    S = {"products": {}, "next_id": 1, "tree": KDTree()}

    def reset_state():
        """Generate the sample products and build the KD-tree from them."""
        prods = generate_products()
        S["products"] = {p["id"]: p for p in prods}
        S["next_id"] = max(S["products"]) + 1
        S["tree"].build([(p["id"], vec_of(p)) for p in prods])
    reset_state()

    def admin_ok():
        """Check the X-Admin-Token header. Uses env var ADMIN_TOKEN, else the dev fallback.
        hmac.compare_digest compares in constant time (safer than ==)."""
        token = os.environ.get("ADMIN_TOKEN") or "dev-admin-token"
        return hmac.compare_digest(request.headers.get("X-Admin-Token", ""), token)

    def err(msg, code):
        """Build a JSON error response with an HTTP status code."""
        return jsonify({"error": msg}), code

    def active_items():
        """All current products as (id, vector) pairs - the input for brute-force search."""
        return [(pid, vec_of(p)) for pid, p in S["products"].items()]

    def metrics(st, ms):
        """Convert the tree's search stats into the field names the frontend displays."""
        return {"time_ms": ms, "nodes_visited": st["visited"], "branches_pruned": st["pruned"],
                "distance_calculations": st["dist_calcs"]}

    def parse_vector(v):
        """Validate a query vector: must be a list of exactly 5 finite numbers."""
        if not isinstance(v, list) or len(v) != 5 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
            raise ValidationError("vector must be 5 finite numbers")
        return [float(x) for x in v]

    def query_from(body):
        """Work out what the search is centred on. Returns (vector, product_id_or_None).
        Either an existing product_id (its vector is used) or a custom 5D vector."""
        if body.get("product_id") is not None:
            pid = body["product_id"]
            if isinstance(pid, bool) or not isinstance(pid, int) or pid not in S["products"]:
                raise ValidationError("unknown product_id")
            return vec_of(S["products"][pid]), pid
        if "vector" in body: return parse_vector(body["vector"]), None
        raise ValidationError("provide product_id or vector")

    def weights_from(body):
        """Read optional per-dimension weights (5 non-negative numbers); default is all 1.0."""
        try: return check_weights(body.get("weights"), 5)
        except (ValueError, TypeError) as e: raise ValidationError(str(e))

    def pack(res):
        """Turn [(id, distance), ...] into [{product: {...}, distance: ...}, ...] for the frontend."""
        return [{"product": S["products"][pid], "distance": d} for pid, d in res if pid in S["products"]]

    # ---------------- frontend ----------------
    @app.get("/")
    def index():
        """Serve the storefront page, so no separate frontend server is needed."""
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/api/health")
    def health():
        """Simple check that the server is alive."""
        return jsonify({"status": "ok", "products": len(S["products"])})

    # ---------------- products ----------------
    @app.get("/api/products")
    def list_products():
        """Return every product (the shop page loads this once at start)."""
        return jsonify({"products": list(S["products"].values()), "count": len(S["products"])})

    @app.get("/api/products/<int:pid>")
    def get_product(pid):
        """Return one product by id, or 404 if it does not exist."""
        p = S["products"].get(pid)
        return jsonify(p) if p else err("product not found", 404)

    @app.post("/api/products")
    def add_product():
        """ADMIN: create a product. 401 = no/wrong token, 422 = invalid data, 409 = duplicate name."""
        if not admin_ok(): return err("admin token required", 401)
        try: p = clean_product(request.get_json(silent=True))
        except ValidationError as e: return err(str(e), 422)
        if any(x["name"].lower() == p["name"].lower() for x in S["products"].values()):
            return err("a product with this name already exists", 409)
        p["id"] = S["next_id"]; S["next_id"] += 1
        S["products"][p["id"]] = p; S["tree"].insert(p["id"], vec_of(p))   # add to the KD-tree too
        return jsonify(p), 201

    @app.put("/api/products/<int:pid>")
    def update_product(pid):
        """ADMIN: update a product. Its vector may change, so the old tree node is deleted and a new one inserted."""
        if not admin_ok(): return err("admin token required", 401)
        old = S["products"].get(pid)
        if not old: return err("product not found", 404)
        try: p = clean_product(request.get_json(silent=True), partial_base=old)
        except ValidationError as e: return err(str(e), 422)
        if any(x["id"] != pid and x["name"].lower() == p["name"].lower() for x in S["products"].values()):
            return err("a product with this name already exists", 409)
        p["id"] = pid; S["products"][pid] = p
        S["tree"].delete(pid); S["tree"].insert(pid, vec_of(p))
        return jsonify(p)

    @app.delete("/api/products/<int:pid>")
    def delete_product(pid):
        """ADMIN: delete a product. In the tree this is a lazy delete (flag only); rebuild() cleans up later."""
        if not admin_ok(): return err("admin token required", 401)
        if pid not in S["products"]: return err("product not found", 404)
        del S["products"][pid]; S["tree"].delete(pid)
        return jsonify({"deleted": pid})

    # ---------------- searches ----------------
    @app.get("/api/recommendations/<int:pid>")
    def recommend(pid):
        """'You may also like': the k products most similar to product pid (found with the KD-tree).
        With ?compare=1 it also runs brute force and reports whether both agree."""
        if pid not in S["products"]: return err("product not found", 404)
        try:
            k = int(request.args.get("k", 5))
            if not 1 <= k <= MAX_K: raise ValueError
        except ValueError: return err(f"k must be an integer between 1 and {MAX_K}", 422)
        q = vec_of(S["products"][pid]); t0 = time.perf_counter()
        res, st = S["tree"].knn(q, k + 1)                       # ask for k+1 because the product itself is nearest
        ms = (time.perf_counter() - t0) * 1000
        res = [(i, d) for i, d in res if i != pid][:k]          # drop the product itself, keep k
        out = {"product_id": pid, "k": k, "results": pack(res), "metrics": metrics(st, ms)}
        if request.args.get("compare") in ("1", "true", "yes"):
            t0 = time.perf_counter()
            bres, bst = brute_knn(active_items(), q, k + 1, [1.0] * 5)    # slow exact baseline
            bms = (time.perf_counter() - t0) * 1000
            bres = [(i, d) for i, d in bres if i != pid][:k]
            out["brute_force"] = {"time_ms": bms, "nodes_visited": bst["visited"],
                                  "distance_calculations": bst["dist_calcs"],
                                  "agrees": [i for i, _ in bres] == [i for i, _ in res]}   # same ids, same order?
        return jsonify(out)

    @app.post("/api/search/knn")
    def search_knn():
        """k-nearest-neighbour search from a product_id or a custom 5D vector, with optional weights.
        The shop page uses this for the 'KD-tree ranked' sort option."""
        body = request.get_json(silent=True) or {}
        try:
            q, pid = query_from(body); w = weights_from(body); k = body.get("k", 5)
            if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
                raise ValidationError(f"k must be an integer between 1 and {MAX_K}")
            t0 = time.perf_counter()
            res, st = S["tree"].knn(q, k + (1 if pid is not None else 0), w)
            ms = (time.perf_counter() - t0) * 1000
        except (ValidationError, ValueError) as e: return err(str(e), 422)
        if pid is not None: res = [(i, d) for i, d in res if i != pid][:k]   # exclude the query product itself
        return jsonify({"results": pack(res), "metrics": metrics(st, ms)})

    @app.post("/api/search/radius")
    def search_radius():
        """Range search: every product within distance `radius` of the query (negative radius -> 422)."""
        body = request.get_json(silent=True) or {}
        try:
            q, pid = query_from(body); w = weights_from(body)
            t0 = time.perf_counter(); res, st = S["tree"].radius(q, body.get("radius"), w)
            ms = (time.perf_counter() - t0) * 1000
        except (ValidationError, ValueError) as e: return err(str(e), 422)
        if pid is not None: res = [(i, d) for i, d in res if i != pid]
        return jsonify({"results": pack(res), "count": len(res), "metrics": metrics(st, ms)})

    # ---------------- tree info ----------------
    @app.get("/api/kdtree/stats")
    def tree_stats():
        """Tree health: height, active nodes, lazily-deleted nodes, plus the ideal (perfectly balanced) height."""
        s = S["tree"].stats(); n = s["active_nodes"]
        s["optimal_height"] = math.ceil(math.log2(n + 1)) if n else 0
        s["dimensions"] = 5
        return jsonify(s)

    @app.get("/api/kdtree/visualization")
    def tree_viz():
        """First 31 tree nodes in breadth-first order (root first, then level by level) for the explorer panel."""
        t = S["tree"]; nodes = []; queue = [(t.root, 0)] if t.root else []
        while queue and len(nodes) < 31:
            n, depth = queue.pop(0)
            nodes.append({"product_id": n.pid, "depth": depth, "axis": n.axis,
                          "axis_name": AXIS_NAMES[n.axis], "split_value": n.vec[n.axis],
                          "deleted": n.deleted, "left": n.left is not None, "right": n.right is not None})
            if n.left: queue.append((n.left, depth + 1))
            if n.right: queue.append((n.right, depth + 1))
        return jsonify({"total_nodes": t.stats()["active_nodes"], "shown": len(nodes), "nodes": nodes})

    @app.post("/api/kdtree/rebuild")
    def rebuild():
        """ADMIN: physically remove lazily-deleted nodes by rebuilding a fresh balanced tree."""
        if not admin_ok(): return err("admin token required", 401)
        S["tree"].rebuild()
        return jsonify(S["tree"].stats())

    @app.get("/api/benchmark")
    def benchmark():
        """Compare KD-tree vs brute force on random 5D points of growing size (100 ... 2500), 20 queries each."""
        rows = []; r = random.Random(7); Q = 20
        for n in (100, 500, 1000, 2500):
            items = [(i, [r.random() for _ in range(5)]) for i in range(n)]
            t = KDTree(); t.build(items); qs = [[r.random() for _ in range(5)] for _ in range(Q)]
            kt = bt = 0.0; vis = dc = 0; agree = True
            for q in qs:
                a = time.perf_counter(); res, st = t.knn(q, 5); kt += time.perf_counter() - a          # time KD-tree
                a = time.perf_counter(); bres, _ = brute_knn(items, q, 5, [1.0] * 5); bt += time.perf_counter() - a   # time brute force
                vis += st["visited"]; dc += st["dist_calcs"]
                agree = agree and [i for i, _ in res] == [i for i, _ in bres]   # both must give identical answers
            rows.append({"n": n, "kdtree_ms_mean": kt / Q * 1000, "brute_ms_mean": bt / Q * 1000,
                         "kd_nodes_visited_mean": vis / Q, "kd_dist_calcs_mean": dc / Q,
                         "brute_dist_calcs": n, "results_agree": agree})
        return jsonify({"benchmark": rows, "queries_per_size": Q,
                        "note": "Random 5D points, k=5. In 5D, pruning is weaker than in low dimensions, so the speed-up grows with N but is not guaranteed on every dataset."})

    return app


# Runs only when you start the file directly: python backend/app.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))      # PORT env var overrides the default 5000
    create_app().run(host="127.0.0.1", port=port, debug=False)
