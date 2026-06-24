"""
FaceFlash — n_bits Analysis: McNemar paired test + Edge cost model

(a) McNemar test: properly compares configs on the SAME queries (paired).
    Reports which bit-width differences are statistically significant.

(b) Edge cost model: scores each (bits, candidates) cell by candidate-reads
    (proxy for mmap I/O cost), producing the server-vs-edge frontier.

Usage:
    .venv/bin/python benchmarks/bench_nbits_analysis.py
"""
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_data():
    embs = np.load(DATA_DIR / "vggface2_100k_embeddings.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "vggface2_100k_labels.npy", allow_pickle=True)
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)
    qi = []
    for indices in label_to_idx.values():
        if len(indices) >= 4:
            qi.extend(indices[-3:])
        elif len(indices) >= 2:
            qi.append(indices[-1])
    qi = np.array(qi)
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])
    return database, queries, gt


def get_per_query_correctness(database, queries, gt, n_bits, n_cands):
    """Return boolean array: correct[i] = True if query i found its NN."""
    pca = PCABinaryQuantizer(n_bits=n_bits)
    pca.fit(database[:min(5000, len(database))])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    correct = np.zeros(len(queries), dtype=bool)
    for i in range(len(queries)):
        topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cands)
        cand_idx = np.asarray(topk).astype(np.intp)
        cosine_sims = database[cand_idx] @ queries[i]
        best_idx = cand_idx[np.argmax(cosine_sims)]
        correct[i] = (best_idx == gt[i])
    return correct


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test for paired binary outcomes.
    Returns: n_a_only, n_b_only, p_value, significant (at α=0.05)

    n_a_only = queries A got right but B got wrong
    n_b_only = queries B got right but A got wrong
    """
    a_not_b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    b_not_a = np.sum(~correct_a & correct_b)  # B right, A wrong

    # McNemar chi-squared (with continuity correction)
    n = a_not_b + b_not_a
    if n == 0:
        return int(a_not_b), int(b_not_a), 1.0, False

    chi2 = (abs(a_not_b - b_not_a) - 1) ** 2 / n

    # p-value from chi2 with 1 df (manual, no scipy needed)
    # Approximation: p ≈ erfc(sqrt(chi2/2))
    from math import erfc, sqrt
    p_value = erfc(sqrt(chi2 / 2))

    return int(a_not_b), int(b_not_a), p_value, p_value < 0.05


def main():
    print("=" * 70)
    print("  FaceFlash — n_bits Paired Analysis (McNemar + Edge Cost Model)")
    print("=" * 70)

    database, queries, gt = load_data()
    n_q = len(queries)
    print(f"  Database: {len(database):,}, Queries: {n_q}")

    bits_list = [128, 256, 384, 512]
    cands_list = [10, 30, 50, 100, 200, 300]

    # ─── Compute per-query correctness for entire grid ──────────────────
    print("\n  Computing per-query results for all (bits, cands) cells...")
    grid = {}  # (bits, cands) -> bool array
    for n_bits in bits_list:
        print(f"    {n_bits} bits...", end=" ", flush=True)
        pca = PCABinaryQuantizer(n_bits=n_bits)
        pca.fit(database[:min(5000, len(database))])
        db_codes = pca.encode(database)
        q_codes = pca.encode(queries)

        for n_cands in cands_list:
            correct = np.zeros(n_q, dtype=bool)
            for i in range(n_q):
                topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cands)
                cand_idx = np.asarray(topk).astype(np.intp)
                cosine_sims = database[cand_idx] @ queries[i]
                best_idx = cand_idx[np.argmax(cosine_sims)]
                correct[i] = (best_idx == gt[i])
            grid[(n_bits, n_cands)] = correct
        print(f"done")

    # ─── (a) McNemar Paired Tests ───────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  (a) McNemar Paired Tests (α=0.05)")
    print(f"  {'='*70}")
    print(f"  Compares adjacent bit-widths at each candidate count.")
    print(f"  'A>B' = A gets more queries right; p<0.05 = significant.\n")

    print(f"  {'Comparison':<20} {'Cands':<8} {'A>B':<6} {'B>A':<6} {'p-value':<10} {'Sig?'}")
    print(f"  {'─'*60}")

    mcnemar_results = []
    for i in range(len(bits_list) - 1):
        bits_a = bits_list[i]
        bits_b = bits_list[i + 1]
        label = f"{bits_b} vs {bits_a}"
        for n_cands in cands_list:
            ca = grid[(bits_a, n_cands)]
            cb = grid[(bits_b, n_cands)]
            a_only, b_only, p, sig = mcnemar_test(ca, cb)
            marker = "YES ★" if sig else "no"
            print(f"  {label:<20} {n_cands:<8} {a_only:<6} {b_only:<6} {p:<10.4f} {marker}")
            mcnemar_results.append({
                "comparison": label,
                "candidates": n_cands,
                "lower_right_higher_wrong": a_only,
                "higher_right_lower_wrong": b_only,
                "p_value": p,
                "significant": sig,
            })
        print()

    # ─── (b) Edge Cost Model ────────────────────────────────────────────
    print(f"\n  {'='*70}")
    print(f"  (b) Edge vs Server Cost Model")
    print(f"  {'='*70}")
    print(f"  Server cost = memory (bits × N / 8)")
    print(f"  Edge cost = candidate reads (each = 1 random flash read of 2KB)")
    print(f"  Both at ≥99% recall.\n")

    print(f"  {'Config':<18} {'Recall':<8} {'Memory':<10} {'Cand reads':<12} {'Best for'}")
    print(f"  {'─'*65}")

    frontier = []
    for n_bits in bits_list:
        bytes_per_vec = n_bits // 8
        memory_mb = len(database) * bytes_per_vec / 1024 / 1024
        for n_cands in cands_list:
            recall = grid[(n_bits, n_cands)].mean()
            if recall >= 0.99:
                config = f"{n_bits}b / {n_cands} cands"
                # Server cost = memory; edge cost = n_cands (reads)
                frontier.append({
                    "config": config,
                    "n_bits": n_bits,
                    "n_cands": n_cands,
                    "recall": float(recall),
                    "memory_mb": memory_mb,
                    "candidate_reads": n_cands,
                })

    # Find Pareto-optimal for server (minimize memory) and edge (minimize reads)
    # Server: at equal recall ≥99%, minimize memory (= minimize bits)
    server_best = min(frontier, key=lambda x: x["memory_mb"])
    # Edge: at equal recall ≥99%, minimize candidate reads (= minimize cands)
    edge_best = min(frontier, key=lambda x: x["candidate_reads"])

    for f in frontier:
        best_for = []
        if f == server_best:
            best_for.append("SERVER ★")
        if f == edge_best:
            best_for.append("EDGE ★")
        marker = " / ".join(best_for) if best_for else ""
        print(f"  {f['config']:<18} {f['recall']*100:<6.1f}% "
              f"{f['memory_mb']:<8.1f}MB  {f['candidate_reads']:<10}  {marker}")

    print(f"  {'─'*65}")
    print(f"\n  DESIGN RULE:")
    print(f"    Server (floats in RAM): {server_best['config']} → "
          f"{server_best['recall']*100:.1f}% @ {server_best['memory_mb']:.1f} MB")
    print(f"    Edge (floats on flash): {edge_best['config']} → "
          f"{edge_best['recall']*100:.1f}% @ {edge_best['memory_mb']:.1f} MB, "
          f"{edge_best['candidate_reads']} flash reads")
    print(f"\n    Bits and candidates are substitutes.")
    print(f"    Server: minimize memory → fewer bits, more candidates.")
    print(f"    Edge:   minimize I/O   → more bits, fewer candidates.")

    # ─── Save ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "n_queries": n_q,
        "mcnemar_tests": mcnemar_results,
        "frontier_configs": frontier,
        "server_optimal": server_best,
        "edge_optimal": edge_best,
    }
    with open(RESULTS_DIR / "bench_nbits_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_nbits_analysis.json'}")


if __name__ == "__main__":
    main()
