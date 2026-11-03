"""BBNet 재현 노트북 v8 — CPU 단계 실행기: python bbstages.py {generate,labels,bench,cert,table7} --config cfg.json
torch 를 import 하지 않는 별도 프로세스에서 돈다 (OR-Tools 와 PyTorch 를 한 프로세스에 올리면 segfault)."""
import argparse, json, math, os, time
import numpy as np
import bbdist
import bbrepro as R


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _paths(cfg, name):
    d = cfg.get("drive_dir")
    return os.path.join(cfg["work_dir"], name), (os.path.join(d, name) if d else None)


def _store(cfg, name):
    p, mirror = _paths(cfg, name)
    return R.JsonlStore(p, mirror, cfg.get("sync_every", 300))


def _save_json(cfg, name, obj):
    for p in _paths(cfg, name):
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(obj, f)


def _load_json(cfg, name):
    for p in _paths(cfg, name):
        if p and os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(name)


class Progress:
    def __init__(self, total, label, every=60):
        self.total, self.label, self.every, self.n, self.t0, self.last = total, label, every, 0, time.time(), 0

    def tick(self, extra=""):
        self.n += 1
        now = time.time()
        if now - self.last > self.every or self.n == self.total:
            el = now - self.t0
            eta = el / self.n * (self.total - self.n)
            log(f"{self.label}: {self.n:,}/{self.total:,}  경과 {el/60:.1f}분  남은 시간 ~{eta/60:.1f}분 {extra}")
            self.last = now


def _run(cfg, store, fn, tasks, timeout, label):
    retry = cfg.get("retry_errors", False)   # 기본: 실패로 끝난 작업은 다시 돌리지 않는다 (같은 입력이면 다시 실패)
    tasks = [(k, a) for k, a in tasks if k not in store or (retry and (store.get(k) or {}).get("status") == "err")]
    log(f"{label}: 남은 작업 {len(tasks):,}개 (이미 완료 {len(store):,}개), 프로세스 {cfg['n_proc']}개")
    if not tasks:
        return
    pr = Progress(len(tasks), label)
    errs = []
    def cb(key, val, status):
        store.put(key, val if status == "ok" else {"status": status, "error": str(val)[:300]})
        if status != "ok" and len(errs) < 3:
            errs.append(key); log(f"  [{status}] {key}: {str(val)[:300]}")
        pr.tick()
    try:
        R.run_pool(fn, tasks, cfg["n_proc"], timeout, cb)
    finally:
        store.sync()


# ------------------------------------------------------------------ 단계들
def stage_generate(cfg):
    store = _store(cfg, "gen_chunks.jsonl")
    exclude = frozenset(R.excluded_keys())
    n_css, n_pbb, chunk = cfg["n_css"], cfg["n_pbb"], cfg.get("gen_chunk", 100)
    quotas = {("css", f): n_css // len(R.CSS_FAMILIES) for f in R.CSS_FAMILIES}
    quotas.update({("pbb", f): n_pbb // len(R.PBB_FAMILIES) for f in R.PBB_FAMILIES})
    qid = {q: i for i, q in enumerate(sorted(quotas))}
    for rnd in range(50):
        seen, have, used = set(), {q: [] for q in quotas}, {q: 0 for q in quotas}
        for key in sorted(store.data, key=lambda k: (k.split("|")[0], k.split("|")[1], int(k.split("|")[2]))):
            kind, fam, idx = key.split("|")
            q = (kind, fam)
            used[q] = max(used[q], int(idx) + 1)
            for r in store.get(key) or []:
                if isinstance(r, dict) and r.get("key") and r["key"] not in seen:
                    seen.add(r["key"]); have[q].append(r)
        deficit = {q: quotas[q] - len(have[q]) for q in quotas}
        log("계열별 확보/목표: " + ", ".join(f"{k}-{f} {len(have[(k, f)])}/{quotas[(k, f)]}" for k, f in sorted(quotas)))
        if all(v <= 0 for v in deficit.values()):
            break
        tasks = []
        for q, dq in deficit.items():
            for j in range(math.ceil(max(dq, 0) * 1.15 / chunk)):
                idx = used[q] + j
                seed = cfg["seed"] * 1_000_003 + qid[q] * 100_000 + idx
                tasks.append((f"{q[0]}|{q[1]}|{idx}", (chunk, q[0] == "pbb", q[1], seed, cfg["n_max"], exclude)))
        _run(cfg, store, R.generate_chunk, tasks, 24 * 3600, f"생성 라운드 {rnd + 1}")
    recs = [r for q in sorted(quotas) for r in have[q][:quotas[q]]]
    _save_json(cfg, "candidates.json", recs)
    log(f"후보 {len(recs):,}개 저장 (CSS {sum(r['C'] == [] for r in recs):,} / PBB {sum(r['C'] != [] for r in recs):,})")


def stage_labels(cfg):
    recs = _load_json(cfg, "candidates.json")
    store = _store(cfg, "labels.jsonl")
    tasks = [(r["key"], (r, cfg["label_preset"], cfg["seed"])) for r in recs]
    _run(cfg, store, R.label_task, tasks, cfg.get("label_timeout", 6 * 3600), "거리 라벨")
    out, dropped = [], 0
    for r in recs:
        lab = store.get(r["key"]) or {}
        d = lab.get("d")
        if d is None or not (R.D_MIN <= d <= R.D_MAX):
            dropped += 1; continue
        out.append(dict(r, d=int(d), label_methods=lab.get("methods")))
    perm = np.random.default_rng(cfg["seed"]).permutation(len(out))
    n_te = n_va = len(out) // 10
    for rank, i in enumerate(perm):
        out[i]["split"] = "test" if rank < n_te else ("val" if rank < n_te + n_va else "train")
    _save_json(cfg, "dataset.json", out)
    log(f"라벨 완료: {len(out):,}개 (범위 밖/실패로 제외 {dropped:,}개) → train/val/test 80:10:10")


def _test_subset(cfg, limit):
    recs = [r for r in _load_json(cfg, "dataset.json") if r["split"] == "test"]
    if limit and limit < len(recs):
        idx = np.random.default_rng(cfg["seed"] + 7).choice(len(recs), limit, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    return recs


def _bench_task(rec, method, trials, seed):
    d, ok, sec = R.cd_distance(R.code_from_record(rec), method, trials, seed)
    return dict(d=d, valid=ok, sec=sec)


def stage_bench(cfg):
    recs = _test_subset(cfg, cfg.get("bench_max"))
    store = _store(cfg, "bench.jsonl")
    tasks = []
    for name in cfg["bench_methods"]:
        meth, trials = R.BENCH_METHODS[name]
        tasks += [(f"{name}|{r['key']}", (r, meth, trials, cfg["seed"])) for r in recs]
    _run(cfg, store, _bench_task, tasks, cfg.get("bench_timeout", 1800), "휴리스틱 벤치마크")


def _cert_task(rec, solver, time_limit):
    if solver == "CP-SAT":
        return R.cpsat_distance(rec, time_limit)
    return R.mip_distance(rec, time_limit, solver)


def stage_cert(cfg):
    recs = _test_subset(cfg, cfg.get("cert_max"))
    store = _store(cfg, "cert.jsonl")
    tl = cfg["cert_time_limit"]
    skip = cfg.get("cert_skip_above")
    if skip:
        n_skip = 0
        for r in recs:
            if r["d"] > skip and r["key"] not in store:
                store.put(r["key"], {"status": "SKIPPED", "d": None, "bound": None, "sec": 0.0}); n_skip += 1
        store.sync()
        log(f"라벨 거리 > {skip} 인 코드 {n_skip:,}개는 인증 생략 (상한 부분집합으로 집계)")
    recs = sorted(recs, key=lambda r: (r["d"], 2 * r["l"] * r["m"]))
    tasks = [(r["key"], (r, cfg.get("cert_solver", "CP-SAT"), tl)) for r in recs]
    _run(cfg, store, _cert_task, tasks, tl * 1.5 + 120, "정확한 거리 인증")


def _t7_task(i, what, seed):
    code, _ = R.table7_codes()[i]
    t0 = time.time()
    if what == "label:RIS":
        tr, pat = _T7_PRESET["ris"]
        d = int(bbdist.code_parameters(code.ell, code.m, code.A, code.B, tr, pat, seed)[2])
        return dict(d=d, valid=True, sec=time.time() - t0)
    if what.startswith("label:"):
        meth = what.split(":", 1)[1]
        d, ok, sec = R.cd_distance(code, meth, _T7_PRESET[meth], seed)
        return dict(d=d, valid=ok, sec=sec)
    meth, trials = R.BENCH_METHODS[what]
    d, ok, sec = R.cd_distance(code, meth, trials, seed)
    return dict(d=d, valid=ok, sec=sec)


_T7_PRESET = R.LABEL_PRESETS["standard"]


def stage_table7(cfg):
    """기준 거리는 방법별 작업으로 나눠 계산한다 (느린 방법 하나가 시간 초과돼도 나머지 결과는 남도록)."""
    global _T7_PRESET
    p = cfg.get("t7_label_preset", cfg["label_preset"])
    _T7_PRESET = R.LABEL_PRESETS[p] if isinstance(p, str) else p
    store = _store(cfg, "table7.jsonl")
    codes = R.table7_codes()
    idx = range(min(len(codes), cfg.get("t7_max") or len(codes)))
    tasks = []
    for i in idx:
        if (store.get(f"{i}|label") or {}).get("d"):   # 이전 버전에서 이미 계산된 기준 거리는 재사용
            continue
        css = codes[i][0].is_css
        if css and _T7_PRESET["ris"][0] > 0:
            tasks.append((f"{i}|label:RIS", (i, "label:RIS", cfg["seed"])))
        # PBB 에서는 BP-OSD 를 뺀다: 매우 느리고, 논문 표 7 에서도 PBB 코드에서는 모두 실패
        tasks += [(f"{i}|label:{m}", (i, f"label:{m}", cfg["seed"]))
                  for m in ("QDistRndMW", "QDistEvol", "decoderDist")
                  if _T7_PRESET.get(m, 0) > 0 and (css or m != "decoderDist")]
    tasks += [(f"{i}|{m}", (i, m, cfg["seed"])) for i in idx
              for m in cfg.get("t7_methods", ["BP-OSD", "QDistEvol", "QDistRnd"])]
    _run(cfg, store, _t7_task, tasks, cfg.get("t7_timeout", 3600), "표 7 코드")


_RELABEL = dict(ris=(300000, 30000), QDistRndMW=10000, QDistEvol=10000)


def _relabel_task(rec, what, seed):
    code = R.code_from_record(rec)
    t0 = time.time()
    if what == "RIS":
        tr, pat = _RELABEL["ris"]
        d = int(bbdist.code_parameters(code.ell, code.m, code.A, code.B, int(tr), int(pat), seed)[2])
        return dict(d=d, valid=True, sec=time.time() - t0)
    d, ok, sec = R.cd_distance(code, what, _RELABEL[what], seed)
    return dict(d=d, valid=ok, sec=sec)


def stage_relabel(cfg):
    """(v7) 지정한 테스트 코드의 거리를 더 큰 예산·다른 시드로 다시 계산 (방법별 작업)."""
    global _RELABEL
    _RELABEL = cfg.get("relabel_preset") or _RELABEL
    groups = _load_json(cfg, "relabel_groups.json")
    keys = set(groups["disagree"]) | set(groups["control"])
    recs = [r for r in _load_json(cfg, "dataset.json") if r["key"] in keys]
    store = _store(cfg, "relabel.jsonl")
    seed = cfg["seed"] + 1000          # 원래 라벨과 다른 시드 (독립적인 탐색)
    tasks = []
    for r in recs:
        if not r["C"] and _RELABEL.get("ris", (0, 0))[0] > 0:
            tasks.append((f"{r['key']}|RIS", (r, "RIS", seed)))
        tasks += [(f"{r['key']}|{m}", (r, m, seed)) for m in ("QDistRndMW", "QDistEvol") if _RELABEL.get(m, 0) > 0]
    _run(cfg, store, _relabel_task, tasks, cfg.get("relabel_timeout", 3600), "라벨 정밀 재계산")


def stage_check(cfg):
    import bbdist
    known = {"[[72,12,6]]": (6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 12, 6),
             "[[90,8,10]]": (15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)], 8, 10),
             "[[144,12,12]]": (12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], 12, 12),
             "[[288,12,18]]": (12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)], 12, 18)}
    log("1) 알려진 BB 코드 (Bravyi et al. 2024): 자체 RIS 와 codeDistance QDistRnd")
    for name, (l, m, A, B, k0, d0) in known.items():
        c = R.make_code(l, m, A, B)
        d1 = bbdist.code_parameters(l, m, c.A, c.B, 30000, 3000)[2]
        d2, ok, sec = R.cd_distance(c, "QDistRndMW", 2000)
        log(f"   {name:>14s} k={c.k} (기대 {k0}) | RIS d={d1} | QDistRnd d={d2} (witness 유효: {ok}) | 기대 d={d0} "
            f"{'✓' if (c.k, d1, d2) == (k0, d0, d0) else '✗'}")
    log("2) 표 7 의 18개 코드: n 일치, k > 0, PBB 교환 조건")
    bad = [i for i, (c, _) in enumerate(R.table7_codes()) if c.k == 0 or not R.commutes(c)]
    log(f"   문제 있는 코드: {bad or '없음'}")
    log("3) 증강 대칭이 k, 교환 조건, 궤도 키, 정확한 거리(CP-SAT 인증)를 보존하는지 (n ≤ 60)")
    rng, nb_struct, nb_dist, n_cmp = np.random.default_rng(1), 0, 0, 0
    for pbb, fam in ((False, "trinomial"), (False, "weight4"), (True, "mixed")):
        for r in R.generate_chunk(2, pbb, fam, seed=11, n_max=60):
            c = R.code_from_record(r)
            e0 = R.cpsat_distance(r, 20)
            for _ in range(1):
                s2 = R.random_symmetry(c, rng)
                c2 = R.make_code(s2.ell, s2.m, s2.A, s2.B, s2.C, s2.D)
                nb_struct += (c2.k != c.k) + (not R.commutes(c2)) + (R.orbit_key(c2) != r["key"])
                e1 = R.cpsat_distance(R.code_to_record(c2), 20)
                if e0["status"] == e1["status"] == "OPTIMAL":
                    n_cmp += 1; nb_dist += e0["d"] != e1["d"]
    log(f"   k/교환/궤도 키 불일치: {nb_struct} | 정확한 거리 비교 {n_cmp}쌍 중 불일치: {nb_dist}")
    log("4) 정확한 거리 인증 (CP-SAT): [[72,12,6]]")
    log(f"   {R.cpsat_distance(R.code_to_record(R.make_code(*known['[[72,12,6]]'][:4])), 60)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "generate", "labels", "bench", "cert", "table7", "relabel"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg.setdefault("n_proc", os.cpu_count())
    R.warmup()
    globals()["stage_" + a.stage](cfg)
