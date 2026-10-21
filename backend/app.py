import os, time, random, statistics, math
from flask import Flask, jsonify, request
from distance import feature_vector, check_weights
from kdtree import KDTree, brute_knn, brute_radius

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev-admin-token")  # Development fallback only; set ADMIN_TOKEN in deployments.

# All product data lives in memory. No SQLite or other database is used.
def make_products(n=160):
    rng = random.Random(42)
    colors = [("Red",0),("Orange",30),("Yellow",55),("Green",120),("Teal",175),("Blue",220),("Navy",240),("Purple",280),("Pink",330)]
    cats = {"T-Shirt":(499,1800,140,200,5,15), "Shirt":(899,3200,100,160,0,8), "Jeans":(1499,5000,300,450,0,4), "Trousers":(1299,4200,180,300,2,10), "Jacket":(2499,7800,250,550,0,6), "Dress":(1199,6000,80,180,3,20)}
    products=[]
    for i in range(n):
        cat=list(cats)[i%len(cats)]; p0,p1,g0,g1,e0,e1=cats[cat]; color,hue=colors[(i//6)%len(colors)]
        products.append({"id":i+1,"name":f"{color} {cat} #{i+1:03d}","description":f"A {color.lower()} {cat.lower()} for everyday wear.","category":cat,"price":round(rng.uniform(p0,p1)),"gsm":round(rng.uniform(g0,g1)),"elasticity":round(rng.uniform(e0,e1),1),"hue":(hue+rng.uniform(-12,12))%360,"color":color,"image":"/placeholder.svg","stock":rng.randint(0,60),"rating":round(rng.uniform(3,5),1)})
    return products

def create_app():
    app=Flask(__name__, static_folder="../frontend", static_url_path="")
    products=make_products(); state={"tree":None,"vecs":{},"next_id":len(products)+1}
    def rows(): return products
    def vec(p): return feature_vector(p["price"],p["gsm"],p["elasticity"],p["hue"])
    def rebuild():
        state["vecs"]={p["id"]:vec(p) for p in products}; tree=KDTree(); tree.build(state["vecs"].items()); state["tree"]=tree
    rebuild()
    def err(message,code=400): return jsonify(error=message),code
    def admin(): return request.headers.get("X-Admin-Token")==ADMIN_TOKEN
    def parse_product(d,partial=False):
        if not isinstance(d,dict): raise ValueError("request body must be a JSON object")
        out={}
        for f in ("name","category"):
            if f in d or not partial:
                if not isinstance(d.get(f),str) or not d[f].strip(): raise ValueError(f"{f} is required")
                if len(d[f].strip()) > (120 if f == "name" else 60): raise ValueError(f"{f} is too long")
                out[f]=d[f].strip()
        for f in ("price","gsm","elasticity","hue","stock","rating"):
            if f in d or (not partial and f in ("price","gsm","elasticity","hue")):
                try: out[f]=float(d[f])
                except (KeyError,TypeError,ValueError): raise ValueError(f"{f} must be a number")
        bounds={"price":(1,10000000),"gsm":(1,2000),"elasticity":(0,100),"hue":(0,360),"stock":(0,1000000),"rating":(0,5)}
        for f,(lo,hi) in bounds.items():
            if f in out:
                if not math.isfinite(out[f]) or not lo <= out[f] <= hi: raise ValueError(f"{f} must be between {lo} and {hi}")
        if "stock" in out:
            if not out["stock"].is_integer(): raise ValueError("stock must be a whole number")
            out["stock"]=int(out["stock"])
        for f in ("description","color","image"):
            if f in d: out[f]=str(d[f])[:2000]
        return out
    @app.get("/api/health")
    def health(): return jsonify(status="ok", storage="in-memory")
    @app.get("/api/products")
    def list_products():
        ps=list(products); a=request.args
        if a.get("category"): ps=[p for p in ps if p["category"]==a["category"]]
        if a.get("q"): ps=[p for p in ps if a["q"].lower() in p["name"].lower()]
        return jsonify(products=ps,total=len(ps))
    @app.get("/api/products/<int:pid>")
    def get_product(pid):
        p=next((p for p in products if p["id"]==pid),None)
        return (jsonify(p),200) if p else err("product not found",404)
    @app.post("/api/products")
    def add():
        if not admin(): return err("admin token required",401)
        try: d=parse_product(request.get_json(silent=True) or {})
        except ValueError as e: return err(str(e),422)
        if any(p["name"].lower()==d["name"].lower() for p in products): return err("product name already exists",409)
        d.setdefault("description",""); d.setdefault("color",""); d.setdefault("image","/placeholder.svg"); d.setdefault("stock",0); d.setdefault("rating",0)
        d["id"]=state["next_id"]; state["next_id"]+=1; products.append(d); rebuild()
        return jsonify(id=d["id"]),201
    @app.put("/api/products/<int:pid>")
    def upd(pid):
        if not admin(): return err("admin token required",401)
        p=next((p for p in products if p["id"]==pid),None)
        if not p: return err("product not found",404)
        try: d=parse_product(request.get_json(silent=True) or {},True)
        except ValueError as e: return err(str(e),422)
        if not d: return err("nothing to update",422)
        if "name" in d and any(x["id"]!=pid and x["name"].lower()==d["name"].lower() for x in products): return err("product name already exists",409)
        p.update(d); rebuild(); return jsonify(updated=pid)
    @app.delete("/api/products/<int:pid>")
    def rm(pid):
        p=next((p for p in products if p["id"]==pid),None)
        if not admin(): return err("admin token required",401)
        if not p: return err("product not found",404)
        products.remove(p); rebuild(); return jsonify(deleted=pid)
    def run(kind,q,param,w,compare,exclude=None):
        tree=state["tree"]; items=[(i,v) for i,v in state["vecs"].items() if i!=exclude]; extra=1 if exclude in state["vecs"] else 0
        t0=time.perf_counter(); res,st=(tree.knn(q,param+extra,w) if kind=="knn" else tree.radius(q,param,w)); ms=(time.perf_counter()-t0)*1000
        res=[r for r in res if r[0]!=exclude]; res=res[:param] if kind=="knn" else res
        weights=check_weights(w) if w is not None else tree.weights; by={p["id"]:p for p in products}
        out={"results":[{"product":by[i],"distance":d,"vector":state["vecs"][i]} for i,d in res],"query_vector":q,"metrics":{"time_ms":ms,"nodes_visited":st["visited"],"branches_pruned":st["pruned"],"distance_calculations":st["dist_calcs"]}}
        if compare:
            t0=time.perf_counter(); b,bs=brute_knn(items,q,param,weights) if kind=="knn" else brute_radius(items,q,param,weights)
            out["brute_force"]={"time_ms":(time.perf_counter()-t0)*1000,"distance_calculations":bs["dist_calcs"],"agrees":[i for i,_ in b]==[r[0] for r in res]}
        return out
    @app.get("/api/recommendations/<int:pid>")
    def rec(pid):
        if pid not in state["vecs"]: return err("product not found",404)
        try: return jsonify(run("knn",state["vecs"][pid],int(request.args.get("k",5)),None,request.args.get("compare")=="1",pid))
        except (ValueError,TypeError) as e: return err(str(e),422)
    def body_search(kind):
        d=request.get_json(silent=True) or {}
        try:
            q=d.get("vector") or (state["vecs"][d["product_id"]] if d.get("product_id") in state["vecs"] else None)
            if q is None or not isinstance(q,(list,tuple)) or len(q)!=5: raise ValueError("provide product_id or a 5-number vector")
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in q): raise ValueError("vector must contain five finite numbers")
            param=d.get("k",5) if kind=="knn" else d.get("radius")
            if kind=="radius" and param is None: raise ValueError("radius is required")
            if kind=="knn":
                if isinstance(param,bool) or not isinstance(param,int) or not 1 <= param <= 500: raise ValueError("k must be an integer between 1 and 500")
            elif isinstance(param,bool) or not isinstance(param,(int,float)) or not math.isfinite(param) or param < 0 or param > 100: raise ValueError("radius must be finite and between 0 and 100")
            return jsonify(run(kind,list(map(float,q)),param,d.get("weights"),bool(d.get("compare")),d.get("product_id")))
        except (ValueError,TypeError) as e: return err(str(e),422)
    app.post("/api/search/knn")(lambda:body_search("knn")); app.add_url_rule("/api/search/radius","radius",lambda:body_search("radius"),methods=["POST"])
    @app.get("/api/kdtree/stats")
    def stats(): return jsonify(**state["tree"].stats(),total_products=len(state["vecs"]))
    @app.get("/api/kdtree/visualization")
    def visualization():
        nodes=[]
        def walk(node,depth=0):
            if node is None or len(nodes)>=31: return
            nodes.append({"product_id":node.pid,"depth":depth,"axis":node.axis,"axis_name":["price","GSM","stretch","hue cos","hue sin"][node.axis],"split_value":node.vec[node.axis],"left":node.left is not None,"right":node.right is not None})
            walk(node.left,depth+1); walk(node.right,depth+1)
        walk(state["tree"].root)
        return jsonify(nodes=nodes,total_nodes=len(state["vecs"]),shown=len(nodes),note="Pre-order sample of the live KD-tree; each node splits on one of five dimensions.")
    @app.post("/api/kdtree/rebuild")
    def rb():
        if not admin(): return err("admin token required",401)
        rebuild(); return jsonify(**state["tree"].stats())
    @app.get("/api/benchmark")
    def bench():
        out=[]
        for n in (100,500,1000,5000):
            rnd=random.Random(n); items=[(i,[rnd.random(),rnd.random(),rnd.random(),math.cos(a:=rnd.random()*6.2832),math.sin(a)]) for i in range(n)]
            tree=KDTree(); tree.build(items); qs=[items[rnd.randrange(n)][1] for _ in range(20)]; kt=[]; bt=[]; r1=r2=[]
            for _ in range(5):
                t=time.perf_counter(); r1=[tree.knn(q,5) for q in qs]; kt.append((time.perf_counter()-t)/20*1000)
                t=time.perf_counter(); r2=[brute_knn(items,q,5,[1]*5) for q in qs]; bt.append((time.perf_counter()-t)/20*1000)
            out.append({"n":n,"kdtree_ms_mean":statistics.mean(kt),"kdtree_ms_stdev":statistics.stdev(kt),"brute_ms_mean":statistics.mean(bt),"brute_ms_stdev":statistics.stdev(bt),"kd_nodes_visited_mean":statistics.mean(x[1]["visited"] for x in r1),"kd_dist_calcs_mean":statistics.mean(x[1]["dist_calcs"] for x in r1),"kd_pruned_mean":statistics.mean(x[1]["pruned"] for x in r1),"brute_dist_calcs":n,"results_agree":all([i for i,_ in x[0]]==[i for i,_ in y[0]] for x,y in zip(r1,r2))})
        return jsonify(benchmark=out,note="KD-tree is not always faster; in 5D with this small dataset it can be slower than a linear scan.")
    @app.get("/")
    def index(): return app.send_static_file("index.html")
    return app

if __name__=="__main__": create_app().run(port=int(os.environ.get("PORT",5000)),debug=True)
