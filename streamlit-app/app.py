"""RAGspector: Streamlit dashboard.
Live mode : runs your ChromaDB stores + OpenRouter LLM (Com_RAG and RAGspector notebooks).
Demo mode : no data folder found, so it runs on a few built-in sample passages.
Run: streamlit run app.py
"""
import html, json, os, re, time
from collections import Counter
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="RAGspector", page_icon="🔍", layout="wide")

DATA = Path(os.getenv("RAGSPECTOR_DATA", Path(__file__).parent / "data"))
WIKI = DATA / "RAGspector"
STORES = {"wiki": (WIKI / "VectorStore", "Wikipedia_20231101_10k", "Wikipedia (RAGspector)"),
          "python": (DATA / "Python" / "VectorStore", "Python_Documentation", "Python docs (Com_RAG)")}
BENCH = WIKI / "data/benchmark/wikipedia_custom_benchmark_v1.jsonl"
PROBES = WIKI / "data/candidates/candidate_probes_v1.jsonl"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
BASE_K, HELD, NORMALIZE_QUERY = 5, 0.5, True   # see notebook cells 40 vs 64 for NORMALIZE_QUERY
LIVE = any(v[0].exists() for v in STORES.values())

FAM = {"query": ("Query robustness", "#FF3D8B", "Question"), "topk": ("Retrieval robustness", "#27D6F5", "Retrieve"),
       "noise": ("Noise robustness", "#FFA132", "Context"), "order": ("Context order", "#B88BFF", "Context"),
       "conflict": ("Conflict robustness", "#7BE04A", "Context")}
FAMMAP = {"query_robustness": "query", "retrieval_robustness": "topk",
          "noise_robustness": "noise", "context_order_robustness": "order"}

DEMO = [("e1", "Eiffel Tower", "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. It was completed in March 1889 as the entrance arch to that year’s World’s Fair."),
        ("e2", "Eiffel Tower", "Gustave Eiffel’s company designed and built the tower. Construction began in 1887 and took just over two years."),
        ("e3", "Eiffel Tower", "At 330 metres, the tower was the tallest man-made structure in the world until the Chrysler Building surpassed it in 1930."),
        ("g1", "Great Barrier Reef", "The Great Barrier Reef lies in the Coral Sea, off the coast of Queensland, Australia."),
        ("g2", "Great Barrier Reef", "It is composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres."),
        ("g3", "Great Barrier Reef", "The reef is visible from outer space and is the world’s largest coral reef system."),
        ("p1", "Photosynthesis", "During photosynthesis, plants use sunlight, water and carbon dioxide to produce glucose and oxygen."),
        ("p2", "Photosynthesis", "The process takes place in chloroplasts, which contain the pigment chlorophyll."),
        ("p3", "Photosynthesis", "Chlorophyll absorbs mostly blue and red light, which is why leaves look green.")]
DEMO = [{"chunk_id": i, "title": t, "text": x} for i, t, x in DEMO]
DCASES = [{"id": "d1", "q": "When was the Eiffel Tower completed?", "ref": "March 1889", "gold": ["e1"], "para": "At what point was the Eiffel Tower finished?"},
          {"id": "d2", "q": "Where is the Great Barrier Reef located?", "ref": "Off the coast of Queensland, Australia", "gold": ["g1"], "para": "In which place can the Great Barrier Reef be found?"},
          {"id": "d3", "q": "What do plants produce during photosynthesis?", "ref": "Glucose and oxygen", "gold": ["p1"], "para": "Which substances result from photosynthesis in plants?"}]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&display=swap');
h1,h2,h3,.big{font-family:'Bricolage Grotesque',sans-serif!important;letter-spacing:-.01em}
h1{font-weight:800!important;font-size:3rem!important;line-height:1.05!important}
.pipe{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:8px 0 26px}
.st-box{background:#22144F;border:1px solid #3C2B78;border-top:5px solid #F3EEFF;border-radius:12px;padding:12px;min-height:90px}
.chip{display:block;margin-top:6px;border-left:5px solid var(--k);border-radius:6px;padding:3px 8px;font-size:13px;background:color-mix(in srgb,var(--k) 18%,transparent)}
.card{background:#22144F;border:1px solid #3C2B78;border-radius:16px;padding:16px 18px;margin-bottom:12px}
.ans{border-left:8px solid #FF3D8B;font-size:1.15rem}
.rk{font:800 1.8rem 'Bricolage Grotesque',sans-serif;color:#27D6F5;margin-right:12px}
.bar{height:8px;border-radius:4px;background:#3C2B78;max-width:240px;margin:6px 0}.bar i{display:block;height:100%;border-radius:4px;background:var(--k,#27D6F5)}
.mu{color:#BBAEE8;font-size:.9rem}
.res{float:right;font-weight:700;font-size:.85rem;border-radius:999px;padding:2px 12px}
.ok{background:#6BE08A33;color:#6BE08A}.bad{background:#FF7A9F33;color:#FF7A9F}.lock{border:1px dashed #BBAEE8;color:#BBAEE8}
@media(max-width:800px){.pipe{grid-template-columns:1fr}}
</style>""", unsafe_allow_html=True)

esc = html.escape
toks = lambda s: set(re.findall(r"[a-z0-9]+", s.lower())) - set("the a an of in is are was were to and or for on at by what how when where which who do does did can with from as it its this that be".split())

@st.cache_resource
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_col(kb):
    import chromadb
    return chromadb.PersistentClient(path=str(STORES[kb][0])).get_collection(STORES[kb][1])

def retrieve(kb, q, k):
    if kb == "demo":
        qt, out = toks(q), []
        for d in DEMO:
            f = len(qt & toks(d["title"] + " " + d["text"])) / max(len(qt), 1)
            out.append({**d, "similarity": 0.3 + 0.65 * f if f else 0.05})
        return sorted(out, key=lambda d: -d["similarity"])[:k]
    vec = embedder().encode(q, normalize_embeddings=NORMALIZE_QUERY).tolist()
    r = get_col(kb).query(query_embeddings=[vec], n_results=k, include=["documents", "metadatas", "distances"])
    return [{"chunk_id": i, "text": d, "similarity": 1 - x, "title": (m or {}).get("title") or (m or {}).get("source") or "Untitled"}
            for i, d, m, x in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

def generate(q, docs):
    if not LIVE:   # simple extractive stand-in so the demo needs no API key
        qt = toks(q)
        return re.split(r"(?<=\.)\s", max(docs, key=lambda d: len(qt & toks(d["text"])))["text"])[0]
    key = st.secrets.get("OPENROUTER_API_KEY", None) if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing (add it to .streamlit/secrets.toml or the environment).")
    from openai import OpenAI
    ctx = "\n\n".join(f"Title: {d['title']}\n{d['text']}" for d in docs)
    out = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1").chat.completions.create(
        model=MODEL, temperature=0.1, max_tokens=1024,
        messages=[{"role": "system", "content": "Use the following context to answer the question concisely."},
                  {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"}])
    return (out.choices[0].message.content or "").strip()

def f1(a, b):
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    c = sum((Counter(ta) & Counter(tb)).values())
    return 0.0 if not c else 2 * (c / len(tb)) * (c / len(ta)) / ((c / len(tb)) + (c / len(ta)))

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

@st.cache_data
def cases():
    if not LIVE:
        return DCASES
    return [{"id": c["case_id"], "q": c["query"], "ref": c["reference_answer"], "gold": c["gold_evidence_ids"]} for c in load(BENCH)]

BKB = "wiki" if LIVE else "demo"

def specs(c):
    if not LIVE:
        base = retrieve("demo", c["q"], BASE_K)
        other = next(d for d in reversed(DEMO) if d["title"] != base[0]["title"])
        return [dict(family="query", subtype="paraphrase", change="Reworded: " + c["para"], q=c["para"]),
                dict(family="topk", subtype="top_k 5 to 3", change="top_k 5 to 3", k=3),
                dict(family="topk", subtype="top_k 5 to 8", change="top_k 5 to 8", k=8),
                dict(family="noise", subtype="single distractor", change="Added one unrelated passage", docs=base + [other]),
                dict(family="order", subtype="reverse order", change="Same chunks, reversed", docs=base[::-1])]
    out = []
    for p in load(PROBES):
        v = p["validity_checks"]
        if p["base_case_id"] != c["id"] or p["family"] not in FAMMAP or not (v.get("is_valid") and v.get("manual_review_approved")):
            continue
        s, lvl = {"family": FAMMAP[p["family"]], "subtype": p["subtype"].replace("_", " ")}, p["intervention_level"]
        if lvl == "query":
            s.update(q=p["modified_query"], change="Reworded: " + p["modified_query"])
        elif lvl == "retriever_config":
            k = p["retrieval_options"]["top_k"]; s.update(k=k, change=f"top_k {BASE_K} to {k}")
        else:
            s.update(docs=[{"title": d["metadata"].get("title", ""), "text": d["text"]} for d in p["context_intervention"]["documents"]],
                     change="Added one distractor passage" if p["family"] == "noise_robustness" else "Same chunks, reversed order")
        out.append(s)
    return out

def run_case(c):
    base = generate(c["q"], retrieve(BKB, c["q"], BASE_K))
    res = []
    for s in specs(c):
        q = s.get("q", c["q"])
        ans = generate(q, s.get("docs") or retrieve(BKB, q, s.get("k", BASE_K)))
        sc = f1(base, ans)
        res.append({**{k: s[k] for k in ("family", "subtype", "change")}, "held": sc >= HELD, "note": f"Answer overlap {sc:.2f}. Probe answer: {ans[:160]}"})
    return base, res

st.session_state.setdefault("runs", {})

# ---------------- UI ----------------
st.markdown("# Find out where your RAG pipeline breaks.")
st.caption("Ask your pipeline a question, then stress it with probes that reword the query, change top-k, add noise or reorder the evidence.")
stages = ["Question", "Retrieve", "Context", "Generate", "Answer"]
st.markdown('<div class="pipe">' + "".join(
    f'<div class="st-box"><b>{s}</b>' + "".join(f'<span class="chip" style="--k:{c}">{n}</span>' for n, c, st_ in FAM.values() if st_ == s) + "</div>"
    for s in stages) + "</div>", unsafe_allow_html=True)
if not LIVE:
    st.info("Demo mode: no data folder found, so this runs on nine sample passages with a simple extractive answerer. Add your vector stores under `data/` for live results.")

t1, t2, t3 = st.tabs(["Playground", "Stress test", "Scoreboard"])

with t1:
    kbs = [k for k in STORES if STORES[k][0].exists()] if LIVE else ["demo"]
    c1, c2, c3 = st.columns([2, 4, 1])
    kb = c1.selectbox("Knowledge base", kbs, format_func=lambda k: STORES[k][2] if k in STORES else "Sample passages (demo)")
    q = c2.text_input("Question", value="" if LIVE else "When was the Eiffel Tower completed?")
    k = c3.selectbox("Retrieve", [3, 5, 8], index=1, format_func=lambda x: f"top {x}")
    if st.button("Run query", type="primary") and q.strip():
        try:
            with st.spinner("Retrieving and generating…"):
                t = time.perf_counter(); docs = retrieve(kb, q, k); ans = generate(q, docs); dt = time.perf_counter() - t
            st.markdown(f'<div class="card ans"><span class="mu">Answer in {dt:.1f}s</span><br>{esc(ans)}</div>', unsafe_allow_html=True)
            st.subheader("Retrieved evidence")
            for n, d in enumerate(docs, 1):
                st.markdown(f'<div class="card"><span class="rk">{n}</span><b>{esc(d["title"])}</b><div class="bar"><i style="width:{max(0, min(1, d["similarity"])) * 100:.0f}%"></i></div>'
                            f'<span class="mu">Similarity {d["similarity"]:.2f}</span><br>{esc(d["text"][:500])}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

with t2:
    cs = cases()
    if not cs:
        st.warning("No benchmark file found at data/RAGspector/data/benchmark/wikipedia_custom_benchmark_v1.jsonl")
    else:
        c = st.selectbox("Benchmark question", cs, format_func=lambda c: c["q"])
        st.caption(f"Reference answer: **{c['ref']}**. A probe is marked Held when its answer overlaps the baseline answer by at least {HELD:.0%}.")
        if st.button("Run all probes", type="primary"):
            try:
                with st.spinner("Running probes. In live mode this makes one LLM call per probe."):
                    st.session_state.runs[c["id"]] = run_case(c)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        if c["id"] in st.session_state.runs:
            base, res = st.session_state.runs[c["id"]]
            st.markdown(f"**Pipeline baseline answer:** {esc(base)}")
            for p in res:
                n, col, _ = FAM[p["family"]]
                st.markdown(f'<div class="card" style="border-left:8px solid {col}"><span class="chip" style="--k:{col};display:inline-block;margin:0 10px 0 0">{n}</span><b>{esc(p["subtype"])}</b>'
                            f'<span class="res {"ok" if p["held"] else "bad"}">{"Held" if p["held"] else "Drifted"}</span><br>{esc(p["change"])}<br><span class="mu">{esc(p["note"])}</span></div>', unsafe_allow_html=True)
            h = sum(p["held"] for p in res)
            st.markdown(f"**{h} of {len(res)} probes held.**" + (" Check the drifted probes first." if h < len(res) else ""))
            st.markdown('<div class="card" style="border-left:8px solid #7BE04A"><b>Verified contradiction</b><span class="res lock">Needs fixture</span><br>Runs only after you add a manually verified contradictory passage.</div>', unsafe_allow_html=True)

with t3:
    cs = cases()
    if cs:
        R = P = M = 0.0
        for c in cs:
            ids = [d["chunk_id"] for d in retrieve(BKB, c["q"], BASE_K)]
            hit = [i for i in ids if i in c["gold"]]
            R += len(hit) / len(c["gold"]); P += len(hit) / len(ids)
            M += next((1 / (n + 1) for n, i in enumerate(ids) if i in c["gold"]), 0)
        n = len(cs)
        fam = {}
        for _, res in st.session_state.runs.values():
            for p in res:
                v = fam.setdefault(p["family"], [0, 0]); v[0] += p["held"]; v[1] += 1
        h, t = sum(v[0] for v in fam.values()), sum(v[1] for v in fam.values())
        m = st.columns(4)
        m[0].metric(f"Recall@{BASE_K}", f"{R / n:.2f}"); m[1].metric(f"Precision@{BASE_K}", f"{P / n:.2f}")
        m[2].metric("MRR", f"{M / n:.2f}"); m[3].metric("Probes held (this session)", f"{h}/{t}" if t else "None run yet")
        st.subheader("Held rate by probe family")
        for key, (name, col, _) in FAM.items():
            v = fam.get(key)
            label = f"{v[0]}/{v[1]}" if v else ("Awaiting fixtures" if key == "conflict" else "Not run yet")
            st.markdown(f'<div style="display:grid;grid-template-columns:200px 1fr 150px;gap:12px;align-items:center;margin:10px 0"><span>{name}</span>'
                        f'<div class="bar" style="--k:{col};max-width:none;height:14px"><i style="width:{(v[0] / v[1] * 100) if v else 0:.0f}%"></i></div><b>{label}</b></div>', unsafe_allow_html=True)
        st.caption(f"Retrieval metrics cover {n} benchmark questions.")