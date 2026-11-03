# BBNet 재현 노트북 v8
"""BB-Net 논문 재현 도구.

코드 구성(CSS BB / 비-CSS PBB, 논문 식 6-7), Aut(Z_l x Z_m) 대칭과 증강(4.3절), 데이터 생성(5절),
거리 라벨(5절), 휴리스틱 벤치마크(6.1절, codeDistance 패키지), MIP 인증(5절·6.2절).
"""
import collections, contextlib, functools, io, json, math, os, time, warnings
import multiprocessing as mp

import numpy as np

import bbdist

D_MIN, D_MAX = 2, 25   # modeling_bbnet 와 동일 (이 모듈은 torch 를 import 하지 않는다: OR-Tools 와 충돌 방지)


@__import__("dataclasses").dataclass(frozen=True)
class BBCode:
    """modeling_bbnet.BBCode 와 같은 필드·속성 (featurise 에 그대로 넣을 수 있다)."""
    ell: int
    m: int
    k: int
    A: tuple
    B: tuple
    C: tuple = ()
    D: tuple = ()

    @property
    def n(self):
        return 2 * self.ell * self.m

    @property
    def is_css(self):
        return not self.C and not self.D

    @property
    def supports(self):
        return tuple(tuple((int(a), int(b)) for a, b in S) for S in (self.A, self.B, self.C, self.D))

# ============================================================== 코드 구성
def reduce_support(S, ell, m):
    """지수를 (l, m)으로 줄이고 겹치는 단항식을 F2에서 소거 (5절)."""
    cnt = collections.Counter((int(a) % ell, int(b) % m) for a, b in S)
    return tuple(sorted(p for p, c in cnt.items() if c % 2))


def polymat(ell, m, S):
    s, t = np.divmod(np.arange(ell * m), m)
    M = np.zeros((ell * m, ell * m), dtype=np.uint8)
    for a, b in S:
        M[np.arange(ell * m), ((s + a) % ell) * m + (t + b) % m] ^= 1
    return M


def stabilizer_matrix(code):
    """식 (7)의 심플렉틱 검사 행렬 [X | Z] = [[A B | C D], [0 0 | B^T A^T]].
    CSS 코드(C = D = 0)이면 diag(H_X, H_Z) 와 같다 (식 6)."""
    l, m = code.ell, code.m
    A, B, C, D = (polymat(l, m, P) for P in (code.A, code.B, code.C, code.D))
    Z = np.zeros_like(A)
    return np.ascontiguousarray(np.vstack([np.hstack([A, B, C, D]), np.hstack([Z, Z, B.T, A.T])]))


def gf2_rank(M):
    return int(bbdist._rank_packed(bbdist.pack(np.ascontiguousarray(M, dtype=np.uint8))))


def logical_dim(code):
    return int(code.n - gf2_rank(stabilizer_matrix(code)))


def pbb_commutes(ell, m, A, B, C, D):
    """첫 블록 행끼리의 교환 조건: A C^T + B D^T 가 대칭  <=>  f = A C* + B D* 의 지지집합이 g -> -g 에 닫힘."""
    cnt = collections.Counter()
    for P, Q in ((A, C), (B, D)):
        for a, b in P:
            for c, d in Q:
                cnt[((a - c) % ell, (b - d) % m)] += 1
    f = {p for p, v in cnt.items() if v % 2}
    return all(((-a) % ell, (-b) % m) in f for a, b in f)


def commutes(code):
    S = stabilizer_matrix(code).astype(np.int64)
    n = code.n
    return not ((S[:, :n] @ S[:, n:].T + S[:, n:] @ S[:, :n].T) % 2).any()


def make_code(ell, m, A, B, C=(), D=()):
    A, B, C, D = (reduce_support(P, ell, m) for P in (A, B, C, D))
    code = BBCode(ell, m, 0, A, B, C, D)
    return BBCode(ell, m, logical_dim(code), A, B, C, D)


def code_from_record(r):
    t = lambda S: tuple(tuple(int(v) for v in p) for p in S)
    return BBCode(int(r["l"]), int(r["m"]), int(r["k"]), t(r["A"]), t(r["B"]), t(r.get("C", ())), t(r.get("D", ())))


def code_to_record(code, **extra):
    r = dict(l=code.ell, m=code.m, k=code.k, A=[list(p) for p in code.A], B=[list(p) for p in code.B],
             C=[list(p) for p in code.C], D=[list(p) for p in code.D])
    r.update(extra)
    return r


def parse_poly(s):
    """'x^8y^2 + x^16y + 1' 같은 문자열 -> 지지집합."""
    out = []
    for term in s.replace(" ", "").split("+"):
        a = b = 0
        if term != "1":
            i = 0
            while i < len(term):
                v, i = term[i], i + 1
                e = ""
                if i < len(term) and term[i] == "^":
                    i += 1
                    while i < len(term) and term[i].isdigit():
                        e, i = e + term[i], i + 1
                e = int(e) if e else 1
                if v == "x": a = e
                elif v == "y": b = e
                else: raise ValueError(term)
        out.append((a, b))
    return out


# ============================================================== Aut(Z_l x Z_m) 와 증강 (4.3절)
@functools.lru_cache(maxsize=None)
def automorphisms(ell, m):
    """Aut(Z_l x Z_m) 전체: (a, b) -> ((p a + q b) mod l, (r a + s b) mod m) 중 전단사인 것.
    q 는 Hom(Z_m, Z_l) (l/g 의 배수), r 은 Hom(Z_l, Z_m) (m/g 의 배수), g = gcd(l, m)."""
    g = math.gcd(ell, m)
    a, b = np.divmod(np.arange(ell * m), m)
    svals = np.arange(m)[:, None]
    out = []
    for p in range(ell):
        for q in range(0, ell, ell // g):
            ia = ((p * a + q * b) % ell) * m
            for r in range(0, m, m // g):
                idx = np.sort(ia[None, :] + (r * a[None, :] + svals * b[None, :]) % m, axis=1)
                ok = (np.diff(idx, axis=1) != 0).all(axis=1)
                out.extend((p, q, r, int(s)) for s in np.flatnonzero(ok))
    return np.array(out, dtype=np.int64)


def apply_aut(S, phi, ell, m):
    p, q, r, s = phi
    return tuple(((p * a + q * b) % ell, (r * a + s * b) % m) for a, b in S)


def _shift(S, t, ell, m):
    return tuple(((a + t[0]) % ell, (b + t[1]) % m) for a, b in S)


def random_symmetry(code, rng):
    """논문 4.3절 증강: 무작위 phi ∈ Aut(G) 를 모든 다항식에 동시 적용,
    CSS: A, B 에 독립 평행이동 + (A, B) -> (B, A);  PBB: 공통 평행이동 + (A,B,C,D) -> (B,A,D,C)."""
    l, m = code.ell, code.m
    auts = automorphisms(l, m)
    phi = auts[rng.integers(len(auts))]
    P = [apply_aut(S, phi, l, m) for S in code.supports]
    rand_t = lambda: (int(rng.integers(l)), int(rng.integers(m)))
    if code.is_css:
        P = [_shift(P[0], rand_t(), l, m), _shift(P[1], rand_t(), l, m), (), ()]
        if rng.random() < 0.5:
            P = [P[1], P[0], (), ()]
    else:
        t = rand_t()
        P = [_shift(S, t, l, m) for S in P]
        if rng.random() < 0.5:
            P = [P[1], P[0], P[3], P[2]]
    return BBCode(l, m, code.k, *P)


def _canon_codes(S, auts, ell, m, anchors=None):
    """모든 phi 에 대해 S 의 이미지를 정렬·정수 인코딩. anchors=None 이면 각자 최소 평행이동(CSS)."""
    base = ell * m + 1
    S = np.asarray(S, dtype=np.int64).reshape(-1, 2)
    a = (auts[:, 0:1] * S[None, :, 0] + auts[:, 1:2] * S[None, :, 1]) % ell
    b = (auts[:, 2:3] * S[None, :, 0] + auts[:, 3:4] * S[None, :, 1]) % m
    def enc(aa, bb):
        v = np.sort(aa * m + bb, axis=1) + 1
        out = np.zeros(len(v), dtype=np.int64)
        for j in range(v.shape[1]):
            out = out * base + v[:, j]
        return out
    if anchors is None:
        return np.min(np.stack([enc((a - a[:, j:j + 1]) % ell, (b - b[:, j:j + 1]) % m)
                                for j in range(S.shape[0])]), axis=0)
    return np.stack([enc((a - an[0][:, None]) % ell, (b - an[1][:, None]) % m) for an in anchors])


def orbit_key(code):
    """증강 대칭군(Aut(G), 평행이동, 레지스터 교환) 궤도의 결정적 대표값 — 대칭 동등 중복 제거용 (5절)."""
    l, m = code.ell, code.m
    auts = automorphisms(l, m)
    big = (l * m + 1) ** 4 + 1
    if code.is_css:
        kA, kB = (_canon_codes(S, auts, l, m) for S in (code.A, code.B))
        return f"C{l},{m}:{int(min((kA * big + kB).min(), (kB * big + kA).min()))}"
    best = None
    for order in ((0, 1, 2, 3), (1, 0, 3, 2)):
        P = [np.asarray(code.supports[i], dtype=np.int64) for i in order]
        A0 = P[0]
        an = [((auts[:, 0] * x + auts[:, 1] * y) % l, (auts[:, 2] * x + auts[:, 3] * y) % m) for x, y in A0]
        ks = [_canon_codes(S, auts, l, m, an) for S in P]          # 각 (anchors, auts)
        hi, lo = ks[0] * big + ks[1], ks[2] * big + ks[3]
        i = np.lexsort((lo.ravel(), hi.ravel()))[0]
        cand = (int(hi.ravel()[i]), int(lo.ravel()[i]))
        best = cand if best is None or cand < best else best
    return f"P{l},{m}:{best[0]}:{best[1]}"


# ============================================================== 데이터 생성 (5절)
def lm_pairs(n_max=360, lo=3, hi=30):
    return [(l, m) for l in range(lo, hi + 1) for m in range(lo, hi + 1) if 2 * l * m <= n_max]


CSS_FAMILIES = ("separable", "swapped", "mixed", "trinomial", "weight4")


def _sample_AB(rng, l, m, fam):
    X = lambda: (int(rng.integers(l)), 0)
    Y = lambda: (0, int(rng.integers(m)))
    M = lambda: (int(rng.integers(1, l)), int(rng.integers(1, m)))
    R = lambda: (int(rng.integers(l)), int(rng.integers(m)))
    if fam == "separable":   # Bravyi et al. 형태  A = x^a + y^b + y^c,  B = y^d + x^e + x^f
        return [X(), Y(), Y()], [Y(), X(), X()]
    if fam == "swapped":     # 좌표 교환형        A = y^a + x^b + x^c,  B = x^d + y^e + y^f
        return [Y(), X(), X()], [X(), Y(), Y()]
    if fam == "mixed":       # 혼합 단항식 x^a y^b (a, b ≠ 0) 포함
        return [R(), M(), M()], [R(), M(), M()]
    if fam == "trinomial":   # 임의의 3항식
        return [R(), R(), R()], [R(), R(), R()]
    if fam == "weight4":     # 4항식 (논문의 'adversarial' 계열 대용, 표 7의 4항 코드 참고)
        return [R(), R(), R(), R()], [R(), R(), R(), R()]
    raise ValueError(fam)


def _find_D(rng, l, m, A, B, C, max_terms=2):
    """주어진 A, B, C 에 대해 교환 조건을 만족하는 1-2항 D 를 전수 탐색해 하나 고른다."""
    G = l * m
    ga, gb = np.divmod(np.arange(G), m)
    neg = ((-ga) % l) * m + (-gb) % m
    def ind(terms):
        v = np.zeros(G, dtype=np.uint8)
        for t in terms: v[t] ^= 1
        return v
    gid = lambda a, b: (a % l) * m + (b % m)
    v1 = ind([gid(a - c, b - d) for a, b in A for c, d in C])
    W = np.zeros((G, G), dtype=np.uint8)           # W[d] = ind(B D*) for D = monomial d
    for g in range(G):
        da, db = divmod(g, m)
        for a, b in B:
            W[g, gid(a - da, b - db)] ^= 1
    cands = []
    V = v1[None, :] ^ W
    ok1 = np.flatnonzero((V == V[:, neg]).all(1))
    cands += [(int(g),) for g in ok1]
    if max_terms >= 2:
        i, j = np.triu_indices(G, 1)
        for s in range(0, len(i), 20000):
            ii, jj = i[s:s + 20000], j[s:s + 20000]
            V = v1[None, :] ^ W[ii] ^ W[jj]
            ok = np.flatnonzero((V == V[:, neg]).all(1))
            cands += [(int(ii[o]), int(jj[o])) for o in ok]
    if not cands:
        return None
    pick = cands[rng.integers(len(cands))]
    return [divmod(g, m) for g in pick]


PBB_FAMILIES = CSS_FAMILIES[:4]   # PBB: 3항 A, B 위에 1-2항 C, D (표 7 참고)


def sample_candidate(rng, pairs, pbb, fam):
    l, m = pairs[rng.integers(len(pairs))]
    A, B = _sample_AB(rng, l, m, fam)
    A, B = reduce_support(A, l, m), reduce_support(B, l, m)
    if len(A) < 2 or len(B) < 2:
        return None
    if not pbb:
        return make_code(l, m, A, B), fam
    C = reduce_support([(int(rng.integers(l)), int(rng.integers(m))) for _ in range(rng.integers(1, 3))], l, m)
    if not C:
        return None
    D = _find_D(rng, l, m, A, B, C)
    if D is None:
        return None
    D = reduce_support(D, l, m)
    if not D or not pbb_commutes(l, m, A, B, C, D):
        return None
    return make_code(l, m, A, B, C, D), "pbb-" + fam


def generate_chunk(n_target, pbb, fam, seed, n_max=360, exclude=frozenset()):
    """계열 fam 에서 k > 0 인 후보 n_target 개 (청크 내부 중복 제거). 반환: 레코드 리스트 (라벨 없음)."""
    rng = np.random.default_rng(seed)
    pairs = lm_pairs(n_max)
    out, seen = [], set()
    while len(out) < n_target:
        res = sample_candidate(rng, pairs, pbb, fam)
        if res is None:
            continue
        code, fam_label = res
        if code.k == 0:
            continue
        key = orbit_key(code)
        if key in seen or key in exclude:
            continue
        seen.add(key)
        out.append(code_to_record(code, fam=fam_label, key=key))
    return out


# ============================================================== 거리 계산: codeDistance 래퍼
BENCH_METHODS = {   # 논문 표 1: (codeDistance 메서드 이름, 코드당 시행 횟수)
    "QDistRnd": ("QDistRndMW", 200),
    "QDistEvol": ("QDistEvol", 200),
    "BP-OSD": ("decoderDist", 200),
    "GEStim": ("GraphLikeErrorStim", 1),
    "CCStim": ("ColourCodeDistStim", 1),
    "UEStim": ("UndetectableErrorStim", 3),
}
STIM_METHODS = {"GraphLikeErrorStim", "ColourCodeDistStim", "UndetectableErrorStim"}


def witness_ok(code, w, d):
    """식 (4): 모든 안정자와 교환, 안정자 군에 속하지 않음, 무게 = d."""
    n = code.n
    w = np.asarray(w, dtype=np.uint8).reshape(-1) % 2
    if w.size != 2 * n or d <= 0:
        return False
    S = stabilizer_matrix(code)
    x, z = w[:n].astype(np.int64), w[n:].astype(np.int64)
    if ((S[:, :n].astype(np.int64) @ z + S[:, n:].astype(np.int64) @ x) % 2).any():
        return False
    if int(np.count_nonzero(x | z)) != d:
        return False
    return gf2_rank(np.vstack([S, w[None, :]])) > gf2_rank(S)


def cd_distance(code, method, trials, seed=0):
    """codeDistance 로 거리 상한 하나. CSS 는 Z 성분만 계산한다:
    BB 코드는 반전 자기동형사상 + 블록 교환으로 X<->Z 가 바뀌므로 항상 d_X = d_Z."""
    import codedistance as cd
    p = {} if method in STIM_METHODS else {"iterCount": int(trials)}
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if code.is_css:
            HX, HZ = bbdist.bb_check_matrices(code.ell, code.m, code.A, code.B)
            res = cd.CSScodeDistance(HX, HZ, method=method, params=p, component="Z", seed=seed)
            L = np.asarray(res["L"], dtype=np.uint8).reshape(-1)
            w = np.concatenate([np.zeros(code.n, np.uint8), L]) if L.size == code.n else L
        else:
            res = cd.codeDistance(stabilizer_matrix(code), tB=2, method=method, params=p, seed=seed)
            w = np.asarray(res["L"], dtype=np.uint8).reshape(-1)
    d = int(res["d"])
    return d, bool(witness_ok(code, w, d)), time.time() - t0


# ============================================================== 라벨 (5절)
LABEL_PRESETS = {
    # ris: 자체 numba 무작위 정보집합 탐색 (CSS 전용, (최대 시행, patience)); 나머지는 codeDistance 시행 횟수
    "fast":     dict(ris=(20000, 2000), QDistRndMW=0,     QDistEvol=1000,  decoderDist=0),
    "standard": dict(ris=(30000, 3000), QDistRndMW=2000,  QDistEvol=2000,  decoderDist=0),
    "paper":    dict(ris=(0, 0),        QDistRndMW=10000, QDistEvol=10000, decoderDist=10000),
}


def label_task(rec, preset, seed=0):
    """여러 휴리스틱 결과 중 유효한 최솟값 = 목표 거리(상한). 방법별 결과도 함께 반환."""
    code = code_from_record(rec)
    cfg = LABEL_PRESETS[preset] if isinstance(preset, str) else preset
    out = {}
    tr, pat = cfg["ris"]
    if code.is_css and tr > 0:
        t0 = time.time()
        out["RIS"] = (int(bbdist.code_parameters(code.ell, code.m, code.A, code.B, tr, pat, seed)[2]), True,
                      time.time() - t0)
    for meth in ("QDistRndMW", "QDistEvol", "decoderDist"):
        if cfg.get(meth, 0) > 0:
            try:
                out[meth] = cd_distance(code, meth, cfg[meth], seed)
            except Exception:
                out[meth] = (0, False, 0.0)
    valid = [d for d, ok, _ in out.values() if ok and d > 0]
    return {"d": int(min(valid)) if valid else None, "methods": out}


# ============================================================== MIP 인증 (5절, 6.2절)
def css_logicals(HX, HZ):
    """X 형 논리 연산자 기저 L_X (ker H_Z / rowspace H_X)."""
    NZ, _ = bbdist.gf2_nullspace(HZ)
    RX, _ = bbdist.gf2_rref(HX)
    return NZ[bbdist._independent_rows(bbdist.pack(RX), bbdist.pack(NZ))]


def symplectic_logicals(S):
    """N(S)/S 의 기저 (2k 개, [X | Z])."""
    n = S.shape[1] // 2
    N, _ = bbdist.gf2_nullspace(np.ascontiguousarray(np.hstack([S[:, n:], S[:, :n]])))
    R, _ = bbdist.gf2_rref(S)
    return N[bbdist._independent_rows(bbdist.pack(R), bbdist.pack(N))]


def cpsat_distance(rec, time_limit=60.0, workers=1):
    """같은 문제를 CP-SAT 로 (패리티 제약을 XOR 로 직접 표현). status == 'OPTIMAL' 이면 인증."""
    from ortools.sat.python import cp_model
    code = code_from_record(rec)
    t0 = time.time()
    M = cp_model.CpModel()
    one = M.NewBoolVar("one"); M.Add(one == 1)
    n = code.n
    if code.is_css:
        HX, HZ = bbdist.bb_check_matrices(code.ell, code.m, code.A, code.B)
        v = [M.NewBoolVar(f"v{i}") for i in range(n)]
        weight, checks, logic = v, HX, css_logicals(HX, HZ)
        prod = lambda row: [v[i] for i in np.flatnonzero(row)]
    else:
        S = stabilizer_matrix(code)
        x = [M.NewBoolVar(f"x{i}") for i in range(n)]
        z = [M.NewBoolVar(f"z{i}") for i in range(n)]
        weight = [M.NewBoolVar(f"w{i}") for i in range(n)]
        for i in range(n):
            M.AddBoolOr([x[i], z[i]]).OnlyEnforceIf(weight[i])
            M.AddImplication(x[i], weight[i]); M.AddImplication(z[i], weight[i])
        checks, logic = S, symplectic_logicals(S)
        prod = lambda row: [z[i] for i in np.flatnonzero(row[:n])] + [x[i] for i in np.flatnonzero(row[n:])]
    for row in checks:
        t = prod(row)
        if t:
            M.AddBoolXOr(t + [one])          # XOR(t) = 0
    ys = []
    for j, row in enumerate(logic):
        y = M.NewBoolVar(f"y{j}")
        M.AddBoolXOr(prod(row) + [y.Not()])   # XOR(t) = y
        ys.append(y)
    M.AddBoolOr(ys)
    M.Minimize(sum(weight))
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = float(time_limit)
    sol.parameters.num_workers = int(workers)
    st = sol.Solve(M)
    name = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}.get(st, "NONE")
    d = int(round(sol.ObjectiveValue())) if name != "NONE" else None
    bound = sol.BestObjectiveBound() if name != "NONE" else None
    return dict(status=name, d=d, bound=bound, sec=time.time() - t0)


def mip_distance(rec, time_limit=60.0, solver_name="SCIP"):
    """정확한 최소 거리 MILP (Punekar et al. 형태, OR-Tools). 반환 dict(status, d, bound, sec).
    status == 'OPTIMAL' 이면 거리가 인증된 것."""
    from ortools.linear_solver import pywraplp
    code = code_from_record(rec)
    t0 = time.time()
    solver = pywraplp.Solver.CreateSolver(solver_name)
    solver.SetNumThreads(1)
    solver.SetTimeLimit(int(time_limit * 1000))
    inf = solver.infinity()
    if code.is_css:      # d = d_Z (BB 코드에서 d_X = d_Z)
        HX, HZ = bbdist.bb_check_matrices(code.ell, code.m, code.A, code.B)
        n = code.n
        v = [solver.BoolVar(f"v{i}") for i in range(n)]
        weight = v
        checks, logic = HX, css_logicals(HX, HZ)
        prod = lambda row: [v[i] for i in np.flatnonzero(row)]
    else:
        S = stabilizer_matrix(code)
        n = code.n
        x = [solver.BoolVar(f"x{i}") for i in range(n)]
        z = [solver.BoolVar(f"z{i}") for i in range(n)]
        weight = [solver.BoolVar(f"w{i}") for i in range(n)]
        for i in range(n):
            solver.Add(weight[i] >= x[i]); solver.Add(weight[i] >= z[i])
        checks, logic = S, symplectic_logicals(S)
        def prod(row):   # 심플렉틱 곱 <row, (x|z)> = row_X . z + row_Z . x
            return [z[i] for i in np.flatnonzero(row[:n])] + [x[i] for i in np.flatnonzero(row[n:])]
    for j, row in enumerate(checks):
        terms = prod(row)
        if terms:
            s = solver.IntVar(0, len(terms) // 2, f"s{j}")
            solver.Add(sum(terms) == 2 * s)
    ys = []
    for j, row in enumerate(logic):
        terms = prod(row)
        y = solver.BoolVar(f"y{j}")
        t = solver.IntVar(0, len(terms) // 2 + 1, f"t{j}")
        solver.Add(sum(terms) == 2 * t + y)
        ys.append(y)
    solver.Add(sum(ys) >= 1)
    solver.Minimize(sum(weight))
    st = solver.Solve()
    name = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(st, "NONE")
    d = int(round(solver.Objective().Value())) if name != "NONE" else None
    bound = solver.Objective().BestBound() if name != "NONE" else None
    return dict(status=name, d=d, bound=bound, sec=time.time() - t0)


# ============================================================== 프로세스 풀 (작업별 타임아웃) + 재개 가능한 저장소
def _pool_worker(conn, fn, args):
    try:
        conn.send(("ok", fn(*args)))
    except BaseException as e:
        conn.send(("err", repr(e)))
    finally:
        conn.close()


def warmup():
    """fork 전에 부모에서 무거운 import 와 numba 컴파일을 끝내 둔다 (자식마다 반복하지 않도록)."""
    import codedistance  # noqa: F401  (qiskit, stim, ortools 등을 함께 불러온다)
    c = make_code(3, 3, [(1, 0), (0, 1), (0, 2)], [(0, 1), (1, 0), (2, 0)])
    bbdist.code_parameters(3, 3, c.A, c.B, 5, 5, 0)
    for l, m in lm_pairs():
        automorphisms(l, m)


def run_pool(fn, tasks, n_proc, timeout, callback):
    """tasks: [(key, args)]. 작업마다 fork 한 프로세스에서 fn(*args) 실행, timeout 초과 시 강제 종료.
    callback(key, result | None, status) 를 메인 프로세스에서 호출 (status: ok / err / timeout)."""
    ctx = mp.get_context("fork")
    pending, running = list(tasks)[::-1], {}
    while pending or running:
        while pending and len(running) < n_proc:
            key, args = pending.pop()
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_pool_worker, args=(w, fn, args), daemon=True)
            p.start(); w.close()
            running[key] = (p, r, time.time())
        done = []
        for key, (p, r, t0) in list(running.items()):
            if r.poll():
                try:
                    status, val = r.recv()
                except EOFError:
                    status, val = "err", None
                p.join(); done.append(key)
                callback(key, val, status)
            elif not p.is_alive():
                p.join(); done.append(key); callback(key, None, "err")
            elif time.time() - t0 > timeout:
                p.kill(); p.join(); done.append(key); callback(key, None, "timeout")
        for key in done:
            running.pop(key)[1].close()
        if not done:
            time.sleep(0.01)


class JsonlStore:
    """key -> 결과 를 로컬 JSONL 에 덧붙이고 주기적으로 Drive 로 복사 (세션이 끊겨도 이어서 실행)."""
    def __init__(self, path, mirror=None, sync_every=300):
        self.path, self.mirror, self.sync_every = path, mirror, sync_every
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if mirror and os.path.exists(mirror) and (not os.path.exists(path) or
                                                  os.path.getsize(mirror) > os.path.getsize(path)):
            import shutil; shutil.copy(mirror, path)
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        k, v = json.loads(line); self.data[k] = v
                    except Exception:
                        pass
        self._f = open(path, "a"); self._last = time.time()

    def __contains__(self, k): return k in self.data
    def __len__(self): return len(self.data)
    def get(self, k, default=None): return self.data.get(k, default)

    def put(self, k, v):
        self.data[k] = v
        self._f.write(json.dumps([k, v]) + "\n")
        if time.time() - self._last > self.sync_every:
            self.sync()

    def sync(self):
        self._f.flush(); self._last = time.time()
        if self.mirror:
            import shutil
            os.makedirs(os.path.dirname(self.mirror), exist_ok=True)
            shutil.copy(self.path, self.mirror)


# ============================================================== 논문 표 7 의 코드 18 개
TABLE7 = [  # (n, l, m), A, B, C, D, GT, BP-OSD, QDistEvol, QDistRnd, BB-Net   (None = Fail)
    ((324, 18, 9), "x^2y^2+x^9y+x^9y^8+x^13y", "x^4y^8+x^8y+x^12y^4+x^16y", "", "", 18, 44, 45, 18, 18),
    ((270, 27, 5), "y^2+x^5+x^16y+x^26y^3", "x^8y^3+x^10y^4+x^24y^3+x^26y^2", "", "", 25, 35, 38, 37, 25),
    ((342, 19, 9), "x^8y^2+x^16y+x^16y^3", "y^5+x^2y^7+x^12y^3", "x^12y^5+x^12y^7", "x^6y^3+x^6y^5", 24, None, 36, 26, 24),
    ((348, 29, 6), "x^8y^2+x^13y+x^25y^3", "x^2y^5+x^12y^4+x^16y^3", "x^11y^4+x^22y", "x^18+x^25y^3", 24, None, 36, 44, 24),
    ((272, 17, 8), "xy^2+x^4y^2+x^8y^3+x^9y^7", "y^4+x^7y^2+x^8y^2+x^12y^2", "", "", 25, 37, 36, 28, 25),
    ((360, 30, 6), "x^2+x^7+x^19y^5", "x^5y^2+x^12y^2+x^26y", "", "", 20, 30, 30, 22, 20),
    ((360, 18, 10), "x^7y^8+x^9y^9+x^14y^2", "y^6+x^10y^8+x^14y^5", "x^5y^8+x^11y^9", "x^9+x^15y^3", 24, None, 34, 42, 24),
    ((336, 28, 6), "x^10y+x^12y^5+x^22", "x^7+x^19y+x^26y^5", "x^15y^2+x^25y^3", "x^5y^3+x^21y^4", 20, None, 30, 28, 20),
    ((336, 28, 6), "x^12+x^16y+x^21y^2", "x^8y^5+x^11+x^27y", "x^12+x^25y^3", "x^11y^3+x^24y^3", 24, None, 34, 28, 24),
    ((216, 18, 6), "x^4y+x^5+x^7y+x^15y", "xy+x^9y^2+x^9y^3+x^13y^2", "", "", 18, 26, 26, 26, 18),
    ((308, 22, 7), "x^9y+x^9y^4+x^12y^2", "x^10y+x^16y^6+x^18y^5", "", "", 20, 20, 28, 20, 20),
    ((250, 25, 5), "x^3+x^6y^3+x^9y^3+x^13y^2", "x^6y^3+x^10y+x^22y+x^22y^2", "", "", 25, 31, 33, 31, 25),
    ((360, 6, 30), "y^4+x^3y^6+x^4", "y^8+y^19+x^3", "", "", 24, 26, 32, 32, 24),
    ((324, 18, 9), "x^5y^5+x^16y^7+x^17y", "x^8y^3+x^10y^3+x^11y^7", "x^11y^3", "x^8y^3+x^13y^7", 20, None, 28, 36, 20),
    ((294, 21, 7), "x^5+x^7y^5+x^9y^5", "x^3y^4+x^7y^5+x^11y^4", "x^5+x^11y^3", "x^5y", 22, None, 30, 28, 22),
    ((350, 25, 7), "x^4y+x^10y^4+x^17y^3", "x^8y^4+x^18y^3+x^19y", "x+x^23y^2", "x^6y+x^21y^4", 24, None, 32, 42, 24),
    ((350, 25, 7), "x^7y^6+x^15+x^18y^2", "x^5y^2+x^18y^3+x^18y^5", "x^23y+x^24y^5", "y^3+x^11y^5", 22, None, 30, 36, 22),
    ((312, 26, 6), "x^7y+x^7y^2+x^24y^3", "x^5y^4+x^10y^5+x^17", "x^14y^3+x^17y^2", "x^12+x^15y^5", 24, None, 32, 34, 24),
]


README_EXAMPLES = {   # modeling_bbnet.EXAMPLES 와 동일
    "[[72,12,6]]": (6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
    "[[90,8,10]]": (15, 3, [(9, 0), (0, 1), (0, 2)], [(0, 0), (2, 0), (7, 0)]),
    "[[144,12,12]]": (12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
}


def excluded_keys():
    """README 예제 3개와 표 7 의 18개 코드 (및 그 대칭 궤도)는 학습·검증·테스트에서 제외."""
    keys = {orbit_key(make_code(*v)) for v in README_EXAMPLES.values()}
    return keys | {orbit_key(c) for c, _ in table7_codes()}


def table7_codes():
    out = []
    for (n, l, m), A, B, C, D, *rest in TABLE7:
        P = [parse_poly(s) if s else [] for s in (A, B, C, D)]
        code = make_code(l, m, *P)
        assert code.n == n
        out.append((code, rest))
    return out
