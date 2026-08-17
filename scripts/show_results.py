import json

with open('data/processed/experiment_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print('--- TOP SIMILARITY SHIFTS (Delta S_ij) ---')
for s in d['top_similarity_shifts'][:8]:
    c1 = s['concept_1']
    c2 = s['concept_2']
    s0 = s['orig_sim']
    st = s['active_sim']
    ds = s['delta_sim']
    print(f"[{c1}] <-> [{c2}] | Orig: {s0:.3f} -> Active: {st:.3f} (Delta: +{ds:.3f})")

print('\n--- SAMPLE HELD-OUT TEST QUERIES ---')
for q in d['all_query_evaluations'][:6]:
    query = q['test_query']
    print(f"\nFuture Test Query: \"{query}\"")
    print("  [Static X0 Top-3]:")
    for r in q['static_top_k'][:3]:
        print(f"    - #{r['rank']} {r['label']} (sim: {r['score']:.3f})")
    print("  [Dynamic X Top-3]:")
    for r in q['dynamic_top_k'][:3]:
        print(f"    - #{r['rank']} {r['label']} (sim: {r['score']:.3f}, mass: {r['mass']:.1f})")
