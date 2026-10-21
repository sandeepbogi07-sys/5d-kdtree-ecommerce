"""Feature encoding and weighted distance."""
import math
# Documented assumptions: fixed ranges used for min-max normalisation, values are clipped.
RANGES = {"price": (300.0, 8000.0), "gsm": (80.0, 600.0), "elasticity": (0.0, 40.0)}

def norm(v, lo, hi):
    if v is None or not math.isfinite(v): raise ValueError("missing/invalid numeric value")
    if hi == lo: return 0.0                      # zero-range safety
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))

def feature_vector(price, gsm, elasticity, hue):
    """[norm_price, norm_gsm, norm_elasticity, cos(hue), sin(hue)]; hue in degrees.
    cos/sin makes 359 deg and 1 deg close (circular hue)."""
    if hue is None or not math.isfinite(hue): raise ValueError("invalid hue")
    a = math.radians(hue % 360)
    return [norm(price, *RANGES["price"]), norm(gsm, *RANGES["gsm"]),
            norm(elasticity, *RANGES["elasticity"]), math.cos(a), math.sin(a)]

def check_weights(w, dim=5):
    if w is None: return [1.0] * dim
    if len(w) != dim or any((not isinstance(x, (int, float))) or not math.isfinite(x) or x < 0 for x in w):
        raise ValueError("weights must be 5 finite non-negative numbers")
    return [float(x) for x in w]

def dist2(q, p, w):
    return sum(wi * (a - b) ** 2 for wi, a, b in zip(w, q, p))

def distance(q, p, w=None):
    return math.sqrt(dist2(q, p, check_weights(w)))
