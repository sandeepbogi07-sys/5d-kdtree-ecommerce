"""From-scratch 5D KD-tree + brute-force baseline (same distance and tie-break: (distance, id))."""
import heapq, math
from distance import dist2, check_weights

class Node:
    __slots__ = ("pid", "vec", "axis", "left", "right", "deleted")
    def __init__(s, pid, vec, axis): s.pid, s.vec, s.axis, s.left, s.right, s.deleted = pid, vec, axis, None, None, False

class KDTree:
    def __init__(self, weights=None, k=5):
        self.k, self.weights, self.root, self.index = k, check_weights(weights, k), None, {}
    # ---- construction: median split, axis = depth mod k (cyclic, as in Bentley 1975) ----
    def build(self, items):
        """items: iterable of (pid, vector). O(n log^2 n) because we sort at each level."""
        self.index = {}
        self.root = self._build(sorted(items, key=lambda t: t[0]), 0)
    def _build(self, items, depth):
        if not items: return None
        ax = depth % self.k
        items = sorted(items, key=lambda t: (t[1][ax], t[0]))
        m = len(items) // 2
        n = Node(items[m][0], items[m][1], ax); self.index[n.pid] = n
        n.left, n.right = self._build(items[:m], depth + 1), self._build(items[m + 1:], depth + 1)
        return n
    # Invariant: left subtree coords <= split, right subtree coords >= split on node.axis.
    def insert(self, pid, vec):
        if pid in self.index and not self.index[pid].deleted: raise ValueError("duplicate id")
        if pid in self.index: self.index[pid].deleted = True   # replaced by new node
        if self.root is None:
            self.root = Node(pid, vec, 0); self.index[pid] = self.root; return
        n = self.root
        while True:
            side = "left" if vec[n.axis] < n.vec[n.axis] else "right"
            c = getattr(n, side)
            if c is None:
                nn = Node(pid, vec, (n.axis + 1) % self.k); setattr(n, side, nn); self.index[pid] = nn; return
            n = c
    def find(self, pid):
        n = self.index.get(pid); return n if n and not n.deleted else None
    def delete(self, pid):
        """Lazy deletion: flag node; call rebuild() to compact."""
        n = self.find(pid)
        if not n: return False
        n.deleted = True; return True
    def rebuild(self):
        self.build([(n.pid, n.vec) for n in self.index.values() if not n.deleted])
    def stats(self):
        def h(n): return 0 if n is None else 1 + max(h(n.left), h(n.right))
        act = sum(1 for n in self.index.values() if not n.deleted)
        return {"height": h(self.root), "active_nodes": act, "deleted_nodes": len(self.index) - act}
    # ---- searches ----
    def knn(self, q, k=1, weights=None):
        """Exact K nearest. Pruning: for a node splitting on axis a with diff=q[a]-split, every point in the
        far subtree has |coord_a - q_a| >= |diff|, so weighted dist^2 >= w[a]*diff^2 (all other terms >= 0
        because weights >= 0). If that lower bound is strictly greater than the current k-th best dist^2,
        the far side cannot contain a better or tied point -> skip. Works for any non-negative weights
        (w[a]=0 gives bound 0, so that branch is never wrongly pruned)."""
        w = check_weights(weights, self.k) if weights is not None else self.weights
        if isinstance(k, bool) or not isinstance(k, int) or k < 1: raise ValueError("k must be integer >= 1")
        st = {"visited": 0, "pruned": 0, "dist_calcs": 0}; heap = []   # entries (-d2, -pid, pid)
        def rec(n):
            if n is None: return
            st["visited"] += 1
            if not n.deleted:
                d = dist2(q, n.vec, w); st["dist_calcs"] += 1
                item = (-d, -n.pid, n.pid)
                if len(heap) < k: heapq.heappush(heap, item)
                elif item > heap[0]: heapq.heapreplace(heap, item)
            diff = q[n.axis] - n.vec[n.axis]
            near, far = (n.left, n.right) if diff < 0 else (n.right, n.left)
            rec(near)
            if far is not None:
                if len(heap) == k and w[n.axis] * diff * diff > -heap[0][0]: st["pruned"] += 1
                else: rec(far)
        rec(self.root)
        res = sorted((-a, pid) for a, _, pid in heap)
        return [(pid, math.sqrt(d)) for d, pid in res], st
    def radius(self, q, r, weights=None):
        w = check_weights(weights, self.k) if weights is not None else self.weights
        if not isinstance(r, (int, float)) or isinstance(r, bool) or not math.isfinite(r) or r < 0: raise ValueError("radius must be finite and >= 0")
        r2 = r * r; out = []; st = {"visited": 0, "pruned": 0, "dist_calcs": 0}
        def rec(n):
            if n is None: return
            st["visited"] += 1
            if not n.deleted:
                d = dist2(q, n.vec, w); st["dist_calcs"] += 1
                if d <= r2: out.append((d, n.pid))
            diff = q[n.axis] - n.vec[n.axis]
            near, far = (n.left, n.right) if diff < 0 else (n.right, n.left)
            rec(near)
            if far is not None:
                if w[n.axis] * diff * diff > r2: st["pruned"] += 1
                else: rec(far)
        rec(self.root)
        return [(pid, math.sqrt(d)) for d, pid in sorted(out)], st

def brute_knn(items, q, k, w):
    w = check_weights(w)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1: raise ValueError("k must be integer >= 1")
    r = sorted((dist2(q, v, w), pid) for pid, v in items)[:k]
    return [(pid, math.sqrt(d)) for d, pid in r], {"visited": len(items), "pruned": 0, "dist_calcs": len(items)}
def brute_radius(items, q, r, w):
    w = check_weights(w)
    if not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0: raise ValueError("radius must be finite and >= 0")
    res = sorted((d, pid) for pid, v in items for d in [dist2(q, v, w)] if d <= r * r)
    return [(pid, math.sqrt(d)) for d, pid in res], {"visited": len(items), "pruned": 0, "dist_calcs": len(items)}
