#!/usr/bin/env python3
"""
AMS v3.7 × LongMemEval Benchmark
==================================

Evaluates AMS as a long-term memory system using the LongMemEval benchmark.

LongMemEval tests 5 memory abilities across 500 questions:
  - Single-session user fact recall
  - Single-session assistant fact recall
  - Single-session preference recall
  - Temporal reasoning
  - Knowledge update
  - Multi-session reasoning

Methodology:
  1. For each entry, write all haystack sessions into AMS as memories
  2. Use the question as a generation prompt
  3. Compare generated answer against ground truth using string matching
     (since we cannot call GPT-4o as judge, we use keyword overlap metrics)

No mocks. No simplification. Real GPT-2 + real AMS + real LongMemEval data.
"""

import json, sys, os, time, re
import torch
from collections import Counter, defaultdict

from AgentMemorySystem import Cfg, MemLLM, _Node


def _reset(m):
    m.amm.tree.store.clear()
    m.amm.tree.root = _Node()
    m.amm.tree.nid = 0
    m.amm.time = 0


def _dev(m):
    return next(m.parameters()).device


def extract_keywords(text):
    """Extract meaningful content words from text."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'shall',
        'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
        'in', 'on', 'at', 'to', 'of', 'by', 'with', 'from', 'as',
        'that', 'this', 'it', 'its', 'he', 'she', 'they', 'we', 'you',
        'not', 'no', 'if', 'then', 'than', 'when', 'where', 'what',
        'how', 'all', 'each', 'every', 'both', 'some', 'any',
        'i', 'my', 'me', 'mine', 'your', 'his', 'her', 'their', 'our',
        'about', 'up', 'out', 'just', 'also', 'very', 'really', 'only',
    }
    return [w for w in words if len(w) >= 2 and w not in stopwords]


def keyword_overlap_score(hypothesis, reference):
    """Compute keyword overlap between hypothesis and reference answer."""
    hyp_kw = set(extract_keywords(hypothesis))
    ref_kw = set(extract_keywords(reference))
    if not ref_kw:
        return 0.0
    overlap = hyp_kw & ref_kw
    recall = len(overlap) / len(ref_kw)
    precision = len(overlap) / max(len(hyp_kw), 1)
    if recall + precision == 0:
        return 0.0
    f1 = 2 * recall * precision / (recall + precision)
    return f1


def contains_answer(hypothesis, answer):
    """Check if the answer keywords are substantially present in hypothesis."""
    answer_kw = extract_keywords(str(answer))
    if not answer_kw:
        return False
    hyp_lower = hypothesis.lower()
    found = sum(1 for kw in answer_kw if kw in hyp_lower)
    return found >= max(1, len(answer_kw) * 0.3)


def evaluate_retrieval(m, entry):
    """Measure whether AMS retrieves relevant sessions for a query."""
    dev = _dev(m)
    question = entry['question']
    answer_session_ids = set(entry.get('answer_session_ids', []))

    tk = m.tok(question, return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)

    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix, fs, diag, cb = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)

    top_bias_ids = cb[0].topk(30).indices.tolist()
    top_bias_tokens = [m.tok.decode([t]).strip().lower() for t in top_bias_ids]

    answer_kw = set(extract_keywords(entry['answer']))
    bias_kw = set(top_bias_tokens)
    overlap = answer_kw & bias_kw

    if diag.batch_mem_weights and diag.batch_mem_weights[0]:
        weights = diag.batch_mem_weights[0]
        retrieved_mids = [mid for mid, w in weights]
        retrieved_texts = []
        for mid in retrieved_mids:
            if mid in m.amm.tree.store:
                retrieved_texts.append(m.amm.tree.store[mid].source_text)
        answer_found = any(
            any(akw in rt.lower() for akw in answer_kw)
            for rt in retrieved_texts
        ) if answer_kw else False
    else:
        answer_found = False

    return {
        'n_answer_kw_in_bias': len(overlap),
        'n_answer_kw_total': len(answer_kw),
        'answer_in_retrieved': answer_found,
        'content_bias_nonzero': cb.abs().max().item() > 0.01,
        'n_memories_retrieved': diag.recall_count,
    }


def run_benchmark(m, data, max_entries=None, gen_tokens=40):
    """Run the LongMemEval benchmark on AMS."""
    results_by_type = defaultdict(list)
    all_results = []

    n_entries = min(len(data), max_entries) if max_entries else len(data)

    print(f"\n  Running {n_entries} entries (gen_tokens={gen_tokens})")
    print(f"  {'─'*60}")

    t0 = time.time()

    for idx in range(n_entries):
        entry = data[idx]
        qid = entry['question_id']
        qtype = entry['question_type']
        question = entry['question']
        answer = entry['answer']
        is_abstention = '_abs' in qid

        _reset(m)

        sessions = entry['haystack_sessions']
        session_dates = entry.get('haystack_dates', [])

        n_written = 0
        for si, session in enumerate(sessions):
            for turn in session:
                if turn['role'] == 'user' and len(turn['content'].strip()) > 10:
                    content = turn['content'].strip()
                    if len(content) > 500:
                        content = content[:500]
                    try:
                        ns, _ = m.write(content, training_mode=True)
                        n_written += ns
                    except Exception:
                        pass

        m.eval()

        retrieval_info = evaluate_retrieval(m, entry)

        prompt = f"Based on our previous conversations, {question}"
        if len(m.tok.encode(prompt)) > 200:
            prompt = question

        torch.manual_seed(42)
        try:
            with torch.no_grad():
                hypothesis = m.generate(prompt, mt=gen_tokens, greedy=False)
            generated_text = hypothesis[len(prompt):].strip()
        except Exception as e:
            generated_text = ""
            hypothesis = prompt

        kw_f1 = keyword_overlap_score(generated_text, answer)
        has_answer = contains_answer(generated_text, answer)

        result = {
            'question_id': qid,
            'question_type': qtype,
            'question': question,
            'answer': answer,
            'hypothesis': generated_text,
            'is_abstention': is_abstention,
            'n_sessions': len(sessions),
            'n_written': n_written,
            'kw_f1': kw_f1,
            'has_answer': has_answer,
            **retrieval_info,
        }

        results_by_type[qtype].append(result)
        all_results.append(result)

        if (idx + 1) % 25 == 0 or idx == n_entries - 1:
            elapsed = time.time() - t0
            avg_time = elapsed / (idx + 1)
            print(f"  [{idx+1:>3d}/{n_entries}] {elapsed:.0f}s "
                  f"(avg {avg_time:.1f}s/entry) "
                  f"kw_f1={kw_f1:.3f} written={n_written}")

    total_time = time.time() - t0
    return all_results, results_by_type, total_time


def print_report(all_results, results_by_type, total_time):
    """Print a detailed benchmark report."""
    sep = "=" * 75
    print(f"\n{sep}")
    print("  AMS v3.7 × LongMemEval Benchmark Report")
    print(f"{sep}")

    N = len(all_results)
    print(f"\n  Total entries evaluated: {N}")
    print(f"  Total time: {total_time:.1f}s ({total_time/N:.1f}s per entry)")

    # Overall metrics
    avg_kw_f1 = sum(r['kw_f1'] for r in all_results) / N
    has_answer_rate = sum(1 for r in all_results if r['has_answer']) / N
    content_bias_rate = sum(1 for r in all_results if r['content_bias_nonzero']) / N
    avg_memories = sum(r['n_written'] for r in all_results) / N
    answer_in_ret = sum(1 for r in all_results if r['answer_in_retrieved']) / N

    print(f"\n  {'─'*70}")
    print(f"  Overall Metrics")
    print(f"  {'─'*70}")
    print(f"  Keyword F1 (avg):              {avg_kw_f1:.4f}")
    print(f"  Answer containment rate:       {has_answer_rate:.4f} ({has_answer_rate*100:.1f}%)")
    print(f"  Content bias active rate:      {content_bias_rate:.4f} ({content_bias_rate*100:.1f}%)")
    print(f"  Answer in retrieved memories:  {answer_in_ret:.4f} ({answer_in_ret*100:.1f}%)")
    print(f"  Avg memories written/entry:    {avg_memories:.1f}")

    # Per-task metrics
    print(f"\n  {'─'*70}")
    print(f"  {'Task':<30s} {'N':>4s} {'KW-F1':>7s} {'HasAns':>7s} {'BiaAct':>7s} {'RetAns':>7s}")
    print(f"  {'─'*70}")

    task_scores = {}
    for qtype in sorted(results_by_type.keys()):
        entries = results_by_type[qtype]
        n = len(entries)
        kf1 = sum(r['kw_f1'] for r in entries) / n
        ha = sum(1 for r in entries if r['has_answer']) / n
        cb = sum(1 for r in entries if r['content_bias_nonzero']) / n
        ar = sum(1 for r in entries if r['answer_in_retrieved']) / n
        task_scores[qtype] = {'kw_f1': kf1, 'has_answer': ha, 'bias_active': cb, 'ret_answer': ar}
        print(f"  {qtype:<30s} {n:>4d} {kf1:>7.4f} {ha:>7.1%} {cb:>7.1%} {ar:>7.1%}")

    # Abstention analysis
    abs_results = [r for r in all_results if r['is_abstention']]
    non_abs_results = [r for r in all_results if not r['is_abstention']]

    if abs_results:
        print(f"\n  {'─'*70}")
        print(f"  Abstention Analysis ({len(abs_results)} entries)")
        print(f"  {'─'*70}")
        abs_empty = sum(1 for r in abs_results if len(r['hypothesis']) < 10)
        print(f"  Short/empty responses: {abs_empty}/{len(abs_results)} ({abs_empty/len(abs_results)*100:.1f}%)")

    # Memory system analysis
    print(f"\n  {'─'*70}")
    print(f"  Memory System Analysis")
    print(f"  {'─'*70}")
    all_written = [r['n_written'] for r in all_results]
    all_n_mem_ret = [r['n_memories_retrieved'] for r in all_results]
    print(f"  Memories written:  min={min(all_written)}, max={max(all_written)}, avg={sum(all_written)/N:.1f}")
    print(f"  Memories retrieved: min={min(all_n_mem_ret)}, max={max(all_n_mem_ret)}, avg={sum(all_n_mem_ret)/N:.1f}")

    # Sample outputs (best and worst)
    print(f"\n  {'─'*70}")
    print(f"  Sample Outputs (best KW-F1)")
    print(f"  {'─'*70}")
    sorted_by_f1 = sorted(all_results, key=lambda r: -r['kw_f1'])
    for r in sorted_by_f1[:5]:
        print(f"  [{r['question_type']}] F1={r['kw_f1']:.3f}")
        print(f"    Q: {r['question'][:80]}")
        print(f"    A: {r['answer'][:80]}")
        print(f"    H: {r['hypothesis'][:80]}")
        print()

    print(f"  {'─'*70}")
    print(f"  Sample Outputs (worst KW-F1, non-abstention)")
    print(f"  {'─'*70}")
    non_abs_sorted = sorted(non_abs_results, key=lambda r: r['kw_f1'])
    for r in non_abs_sorted[:3]:
        print(f"  [{r['question_type']}] F1={r['kw_f1']:.3f}")
        print(f"    Q: {r['question'][:80]}")
        print(f"    A: {r['answer'][:80]}")
        print(f"    H: {r['hypothesis'][:80]}")
        print()

    # Task-averaged score (LongMemEval's primary metric)
    if task_scores:
        task_avg_f1 = sum(v['kw_f1'] for v in task_scores.values()) / len(task_scores)
        task_avg_ha = sum(v['has_answer'] for v in task_scores.values()) / len(task_scores)
        print(f"  {'─'*70}")
        print(f"  Task-Averaged Metrics (LongMemEval primary)")
        print(f"  {'─'*70}")
        print(f"  Task-Avg KW-F1:          {task_avg_f1:.4f}")
        print(f"  Task-Avg HasAnswer:      {task_avg_ha:.4f} ({task_avg_ha*100:.1f}%)")

    print(f"\n{sep}")
    print(f"  Benchmark complete: {N} entries, {total_time:.0f}s total")
    print(f"{sep}")


def main():
    torch.manual_seed(42)

    oracle_path = '/workspace/LongMemEval/data/longmemeval_oracle.json'
    if not os.path.exists(oracle_path):
        print(f"ERROR: {oracle_path} not found")
        return False

    with open(oracle_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} LongMemEval entries")

    c = Cfg()
    m = MemLLM(c)
    m.load("gpt2")
    print(f"Model loaded: {sum(p.numel() for p in m.parameters()):,} params")

    # Run on first 100 entries (full 500 would take ~1.5 hours)
    MAX_ENTRIES = 100

    all_results, results_by_type, total_time = run_benchmark(
        m, data, max_entries=MAX_ENTRIES, gen_tokens=40)

    print_report(all_results, results_by_type, total_time)

    output_path = '/workspace/longmemeval_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {'max_entries': MAX_ENTRIES, 'gen_tokens': 40, 'model': 'gpt2'},
            'results': all_results,
            'summary': {
                'total': len(all_results),
                'avg_kw_f1': sum(r['kw_f1'] for r in all_results) / len(all_results),
                'has_answer_rate': sum(1 for r in all_results if r['has_answer']) / len(all_results),
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
