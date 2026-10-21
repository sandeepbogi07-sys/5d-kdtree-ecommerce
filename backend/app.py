"""From-scratch 5D KD-tree + brute-force baseline (same distance and tie-break: (distance, id)).

FILE OVERVIEW (kdtree.py)
-------------------------
A KD-tree (k-dimensional tree) is a binary tree that stores points in k dimensions.
Here every product is a 5D point:
    [normalised price, normalised GSM, normalised stretch, cos(hue), sin(hue)]
Each node splits the space on ONE axis (dimension). The axis rotates with depth
(depth 0 -> axis 0, depth 1 -> axis 1, ... depth 5 -> axis 0 again).
This lets us find the nearest products WITHOUT checking every product, because
whole branches that cannot contain a closer point are skipped ("pruned").

What is in this file:
    Node          - one tree node (one product)
    KDTree        - build, insert, find, delete (lazy), rebuild, stats, knn, radius
    brute_knn     - slow baseline: checks every product (used to verify and benchmark)
    brute_radius  - slow baseline for radius search
"""
import heapq, math
# dist2 = weighted SQUARED distance; check_weights = validates/defaults the 5 weights.
from distance import dist2, check_weights


class Node:
    """One node of the KD-tree = one product.

    Fields:
        pid     - product id
        vec     - the 5D feature vector of the product
        axis    - which dimension (0..4) this node splits on
        left    - subtree with smaller coordinates on `axis`
        right   - subtree with larger/equal coordinates on `axis`
        deleted - lazy-deletion flag (True = logically removed, still in the tree)
    """
    # __slots__ stores only these fields (no per-object dict) -> saves memory.
    __slots__ = ("pid", "vec", "axis", "left", "right", "deleted")
    # Constructor (written with `s` instead of `self`): children start empty, not deleted.
    def __init__(s, pid, vec, axis): s.pid, s.vec, s.axis, s.left, s.right, s.deleted = pid, vec, axis, None, None, False


class KDTree:
    def __init__(self, weights=None, k=5):
        """Create an empty tree.
        weights: how important each dimension is in the distance (default all 1.0).
        k: number of dimensions (5 for this project).
        self.index maps product id -> Node, so we can find any node in O(1)."""
        self.k, self.weights, self.root, self.index = k, check_weights(weights, k), None, {}

    # ---- construction: median split, axis = depth mod k (cyclic, as in Bentley 1975) ----
    def build(self, items):
        """items: iterable of (pid, vector). O(n log^2 n) because we sort at each level.

        Builds a balanced tree from scratch. Sorting by id first makes the result
        deterministic (same input -> same tree)."""
        self.index = {}   # forget any old nodes
        self.root = self._build(sorted(items, key=lambda t: t[0]), 0)

    def _build(self, items, depth):
        """Recursive helper for build(). Returns the root Node of the subtree made from `items`.

        Steps: pick the axis for this depth, sort by that axis, take the MEDIAN as the
        node, and recurse on the left half and right half. Choosing the median keeps
        the tree balanced, so its height is about log2(n)."""
        if not items: return None                      # base case: no points -> empty subtree
        ax = depth % self.k                            # axis rotates: 0,1,2,3,4,0,1,...
        # Sort by this axis; ties broken by product id so the order is always the same.
        items = sorted(items, key=lambda t: (t[1][ax], t[0]))
        m = len(items) // 2                            # index of the median point
        n = Node(items[m][0], items[m][1], ax); self.index[n.pid] = n   # median becomes this node
        # Points before the median go left, points after go right.
        n.left, n.right = self._build(items[:m], depth + 1), self._build(items[m + 1:], depth + 1)
        return n

    # Invariant: left subtree coords <= split, right subtree coords >= split on node.axis.
    def insert(self, pid, vec):
        """Add one product to the tree without rebuilding. O(log n) average, O(n) worst case
        (worst case happens if the tree becomes unbalanced after many inserts)."""
        # Reject an id that is already active in the tree.
        if pid in self.index and not self.index[pid].deleted: raise ValueError("duplicate id")
        # If the id exists but was lazily deleted, the old node stays flagged deleted and a fresh node replaces it.
        if pid in self.index: self.index[pid].deleted = True   # replaced by new node
        # Empty tree: the new node simply becomes the root, splitting on axis 0.
        if self.root is None:
            self.root = Node(pid, vec, 0); self.index[pid] = self.root; return
        n = self.root
        while True:
            # At each node compare ONE coordinate (the node's axis): smaller goes left, else right.
            side = "left" if vec[n.axis] < n.vec[n.axis] else "right"
            c = getattr(n, side)
            if c is None:
                # Found an empty slot: attach the new node here. Its axis is the NEXT axis (cyclic).
                nn = Node(pid, vec, (n.axis + 1) % self.k); setattr(n, side, nn); self.index[pid] = nn; return
            n = c   # otherwise keep walking down

    def find(self, pid):
        """Return the Node for this product id, or None if it does not exist or is deleted. O(1) via the index dict."""
        n = self.index.get(pid); return n if n and not n.deleted else None

    def delete(self, pid):
        """Lazy deletion: flag node; call rebuild() to compact.

        The node stays in the tree (so the tree structure is not damaged) but searches
        skip it. Returns True if something was deleted, False if the id was not found."""
        n = self.find(pid)
        if not n: return False
        n.deleted = True; return True

    def rebuild(self):
        """Physically remove deleted nodes: collect all ACTIVE (pid, vec) pairs and build a fresh balanced tree."""
        self.build([(n.pid, n.vec) for n in self.index.values() if not n.deleted])

    def stats(self):
        """Return tree health info: height, active node count, deleted (lazy) node count.
        Shown by /api/kdtree/stats. A big 'deleted_nodes' number means rebuild() is due."""
        def h(n): return 0 if n is None else 1 + max(h(n.left), h(n.right))   # height = longest root-to-leaf path
        act = sum(1 for n in self.index.values() if not n.deleted)
        return {"height": h(self.root), "active_nodes": act, "deleted_nodes": len(self.index) - act}

    # ---- searches ----
    def knn(self, q, k=1, weights=None):
        """Exact K nearest. Pruning: for a node splitting on axis a with diff=q[a]-split, every point in the
        far subtree has |coord_a - q_a| >= |diff|, so weighted dist^2 >= w[a]*diff^2 (all other terms >= 0
        because weights >= 0). If that lower bound is strictly greater than the current k-th best dist^2,
        the far side cannot contain a better or tied point -> skip. Works for any non-negative weights
        (w[a]=0 gives bound 0, so that branch is never wrongly pruned).

        Parameters: q = query vector, k = how many neighbours, weights = optional per-dimension weights.
        Returns: (list of (pid, distance) nearest-first, stats dict with visited/pruned/dist_calcs).
        Average O(log n), worst case O(n) (in 5D pruning is weaker, so it can approach O(n))."""
        # Use the weights given, otherwise the tree's default weights.
        w = check_weights(weights, self.k) if weights is not None else self.weights
        if isinstance(k, bool) or not isinstance(k, int) or k < 1: raise ValueError("k must be integer >= 1")
        # st counts work done (shown in the website's metrics). heap holds the best k so far.
        st = {"visited": 0, "pruned": 0, "dist_calcs": 0}; heap = []   # entries (-d2, -pid, pid)
        # heapq is a MIN-heap, so distances are stored NEGATED: heap[0] is then the WORST of the best k.
        def rec(n):
            if n is None: return
            st["visited"] += 1
            if not n.deleted:                                  # lazily deleted nodes are skipped
                d = dist2(q, n.vec, w); st["dist_calcs"] += 1  # squared distance (no sqrt needed to compare)
                item = (-d, -n.pid, n.pid)
                if len(heap) < k: heapq.heappush(heap, item)           # still room: just add
                elif item > heap[0]: heapq.heapreplace(heap, item)     # closer than current worst: replace it
            # diff = how far the query is from this node's splitting plane on this node's axis.
            diff = q[n.axis] - n.vec[n.axis]
            # The NEAR side is the side the query lies on; the FAR side is the other one.
            near, far = (n.left, n.right) if diff < 0 else (n.right, n.left)
            rec(near)                                           # always search the near side first
            if far is not None:
                # Prune check: if the plane itself is already farther than our k-th best,
                # nothing on the far side can beat it -> skip the whole far subtree.
                if len(heap) == k and w[n.axis] * diff * diff > -heap[0][0]: st["pruned"] += 1
                else: rec(far)
        rec(self.root)
        # Sort the heap into nearest-first order, then take sqrt to give real distances.
        res = sorted((-a, pid) for a, _, pid in heap)
        return [(pid, math.sqrt(d)) for d, pid in res], st

    def radius(self, q, r, weights=None):
        """Range search: return ALL products within distance r of query q (nearest first).
        Same idea as knn(), but the pruning limit is the fixed radius instead of the k-th best.
        Returns: (list of (pid, distance), stats dict)."""
        w = check_weights(weights, self.k) if weights is not None else self.weights
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not math.isfinite(r) or r < 0: raise ValueError("radius must be finite and >= 0")
        r2 = r * r; out = []; st = {"visited": 0, "pruned": 0, "dist_calcs": 0}   # compare squared distances, avoids sqrt
        def rec(n):
            if n is None: return
            st["visited"] += 1
            if not n.deleted:
                d = dist2(q, n.vec, w); st["dist_calcs"] += 1
                if d <= r2: out.append((d, n.pid))             # inside the radius -> keep it
            diff = q[n.axis] - n.vec[n.axis]
            near, far = (n.left, n.right) if diff < 0 else (n.right, n.left)
            rec(near)
            if far is not None:
                # If the splitting plane is farther than r, the far side cannot contain any match.
                if w[n.axis] * diff * diff > r2: st["pruned"] += 1
                else: rec(far)
        rec(self.root)
        return [(pid, math.sqrt(d)) for d, pid in sorted(out)], st


# ---- brute-force baselines: check EVERY product. Simple and always correct, so we use them
# ---- to verify the KD-tree's answers ("agrees" flag) and to benchmark speed.
def brute_knn(items, q, k, w):
    """Brute-force k nearest neighbours. items = list of (pid, vector). O(n*d) + sorting.
    Returns the same format as KDTree.knn(): (list of (pid, distance), stats)."""
    w = check_weights(w)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1: raise ValueError("k must be integer >= 1")
    # Compute every distance, sort by (distance, id), keep the first k.
    r = sorted((dist2(q, v, w), pid) for pid, v in items)[:k]
    return [(pid, math.sqrt(d)) for d, pid in r], {"visited": len(items), "pruned": 0, "dist_calcs": len(items)}
def brute_radius(items, q, r, w):
    """Brute-force radius search: every product within distance r of q. Same return format as KDTree.radius()."""
    w = check_weights(w)
    if not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0: raise ValueError("radius must be finite and >= 0")
    # Keep only the points whose squared distance is within r*r, sorted nearest first.
    res = sorted((d, pid) for pid, v in items for d in [dist2(q, v, w)] if d <= r * r)
    return [(pid, math.sqrt(d)) for d, pid in res], {"visited": len(items), "pruned": 0, "dist_calcs": len(items)}
