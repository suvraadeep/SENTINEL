import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def memory_curator(state: SentinelState = None) -> dict:
    """Nightly: cluster L2 → merge duplicates → promote to L4 → extract L3 rules."""
    print("[MemoryCurator] Starting nightly consolidation...")

    if l2_collection.count() == 0:
        print("  L2 empty.")
        return {}

    all_eps   = l2_collection.get(include=["documents","metadatas","embeddings"])
    docs      = all_eps["documents"]
    metas     = all_eps["metadatas"]
    ids       = all_eps["ids"]
    embs      = np.array(all_eps["embeddings"])

    print(f"  L2 episodes to process: {len(docs)}")

    if len(docs) > 1:
        sim = cosine_similarity(embs)
        THRESHOLD = 0.90
        merged, clusters = set(), []
        for i in range(len(docs)):
            if i in merged:
                continue
            cluster = [i]
            for j in range(i+1, len(docs)):
                if j not in merged and sim[i,j] >= THRESHOLD:
                    cluster.append(j)
                    merged.add(j)
            clusters.append(cluster)

        deleted = 0
        for cluster in clusters:
            if len(cluster) > 1:
                scores = [float(metas[i].get("score", 1.0)) for i in cluster]
                best   = cluster[np.argmax(scores)]
                to_del = [ids[i] for i in cluster if i != best]
                l2_collection.delete(ids=to_del)
                deleted += len(to_del)
        print(f"  Merged: {deleted} duplicate episodes removed")

    promoted = 0
    for doc, meta in zip(docs, metas):
        if float(meta.get("score", 1.0)) >= 0.9:
            sql = meta.get("sql", "")
            if sql and "/* " not in sql and len(sql) > 20:
                pt = call_llm(f"In ≤6 words, what SQL problem type: {doc}",
                              model=FAST_MODEL, temperature=0.0)
                l4_store(pt.strip()[:50], sql, doc[:200])
                promoted += 1
    print(f"  Promoted {promoted} episodes → L4")

    summaries = [m.get("result_summary","") for m in metas if m.get("result_summary")]
    if summaries:
        raw = call_llm(
            f"From these summaries, extract 1-3 factual business rules as JSON array:\n"
            f"{chr(10).join(summaries[:8])}",
            model=FAST_MODEL, temperature=0.0
        )
        rules = extract_json(raw, fallback=[])
        if isinstance(rules, list):
            for rule in rules:
                rule_str = str(rule)
                nid = f"rule_{hashlib.md5(rule_str.encode()).hexdigest()[:8]}"
                l3_graph.add_node(nid, type="business_rule", description=rule_str)
            print(f"  Extracted {len(rules)} rules → L3")

    print(f"\n[MemoryCurator] L2:{l2_collection.count()} L4:{l4_collection.count()} "
          f"L3:{l3_graph.number_of_nodes()} nodes")
    return {}
