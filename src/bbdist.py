# BBNet 재현 노트북 v8
"""CSS BB 코드의 k 와 거리 d 의 휴리스틱 상한 (GF(2) 선형대수 + numba).

d_Z = min{ |v| : H_X v = 0,  v ∉ rowspace(H_Z) }  (d_X 도 대칭적으로),  d = min(d_X, d_Z).
ker(H_X) 의 생성행렬을 무작위 열 순서로 기약행사다리꼴로 만든 뒤, 각 행(무게 1 정보집합)과
두 행의 합(Lee–Brickell, p = 2) 중 논리 연산자와 반교환하는 가장 가벼운 벡터를 찾는다.
찾은 값은 항상 실제 거리 이상인 '상한'이며, 시행 횟수가 늘수록 실제 거리에 수렴한다.
"""
import numpy as np
import numba as nb


# ---------------------------------------------------------------- 패리티 검사 행렬
def bb_check_matrices(ell, m, A, B):
    """Bravyi et al. (2024) 규약: x = S_l ⊗ I_m, y = I_l ⊗ S_m,
    H_X = [A | B],  H_Z = [B^T | A^T].  행/열 인덱스는 (s, t) -> s*m + t."""
    s, t = np.divmod(np.arange(ell * m), m)
    def poly(S):
        M = np.zeros((ell * m, ell * m), dtype=np.uint8)
        for a, b in S:
            M[np.arange(ell * m), ((s + a) % ell) * m + (t + b) % m] ^= 1
        return M
    Am, Bm = poly(A), poly(B)
    HX = np.hstack([Am, Bm])
    HZ = np.hstack([Bm.T, Am.T])
    return np.ascontiguousarray(HX), np.ascontiguousarray(HZ)


# ---------------------------------------------------------------- GF(2) 기본 연산
@nb.njit(cache=True)
def gf2_rref(M):
    R = M.copy()
    rows, cols = R.shape
    pivots = np.empty(min(rows, cols), dtype=np.int64)
    r = 0
    for c in range(cols):
        if r == rows:
            break
        p = -1
        for i in range(r, rows):
            if R[i, c]:
                p = i
                break
        if p < 0:
            continue
        if p != r:
            for j in range(cols):
                t = R[r, j]; R[r, j] = R[p, j]; R[p, j] = t
        for i in range(rows):
            if i != r and R[i, c]:
                for j in range(c, cols):
                    R[i, j] ^= R[r, j]
        pivots[r] = c
        r += 1
    return R[:r], pivots[:r]


def gf2_nullspace(M):
    R, piv = gf2_rref(M)
    n = M.shape[1]
    free = np.setdiff1d(np.arange(n), piv)
    N = np.zeros((len(free), n), dtype=np.uint8)
    for t, f in enumerate(free):
        N[t, f] = 1
        N[t, piv] = R[:, f]
    return N, len(piv)


def pack(M):
    """(r, n) 0/1 행렬 -> (r, ceil(n/64)) uint64 비트 패킹 (비트 j -> 워드 j//64의 j%64번째 비트)."""
    r, n = M.shape
    W = (n + 63) // 64
    by = np.packbits(M.astype(np.uint8), axis=1, bitorder="little")
    out = np.zeros((r, W * 8), dtype=np.uint8)
    out[:, :by.shape[1]] = by
    return np.ascontiguousarray(out).view("<u8").astype(np.uint64)


@nb.njit(cache=True)
def _rank_packed(P):
    R = P.copy()
    rows, W = R.shape
    n = W * 64
    r = 0
    for c in range(n):
        if r == rows:
            break
        w_, b_ = c >> 6, np.uint64(c & 63)
        p = -1
        for i in range(r, rows):
            if (R[i, w_] >> b_) & np.uint64(1):
                p = i
                break
        if p < 0:
            continue
        if p != r:
            for w in range(W):
                t = R[r, w]; R[r, w] = R[p, w]; R[p, w] = t
        for i in range(r + 1, rows):
            if (R[i, w_] >> b_) & np.uint64(1):
                for w in range(w_, W):
                    R[i, w] ^= R[r, w]
        r += 1
    return r


def logical_dimension(ell, m, A, B):
    """k = n - rank(H_X) - rank(H_Z)  (BB 코드에서는 rank(H_X) = rank(H_Z))."""
    HX, HZ = bb_check_matrices(ell, m, A, B)
    return HX.shape[1] - _rank_packed(pack(HX)) - _rank_packed(pack(HZ))


@nb.njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@nb.njit(cache=True)
def _independent_rows(Bp, Vp):
    """span(Bp)에 속하지 않는 Vp의 행 인덱스(서로 독립)를 반환."""
    W = Bp.shape[1]
    nb_ = Bp.shape[0] + Vp.shape[0]
    basis = np.zeros((nb_, W), dtype=np.uint64)
    piv = np.zeros(nb_, dtype=np.int64)
    nbasis = 0
    keep = np.empty(Vp.shape[0], dtype=np.int64)
    nkeep = 0
    for src in range(2):
        rows = Bp if src == 0 else Vp
        for i in range(rows.shape[0]):
            v = rows[i].copy()
            for t in range(nbasis):
                p = piv[t]
                if (v[p >> 6] >> np.uint64(p & 63)) & np.uint64(1):
                    for w in range(W):
                        v[w] ^= basis[t, w]
            lead = -1
            for w in range(W):
                if v[w] != 0:
                    x = v[w]
                    b = 0
                    while not ((x >> np.uint64(b)) & np.uint64(1)):
                        b += 1
                    lead = w * 64 + b
                    break
            if lead >= 0:
                basis[nbasis] = v
                piv[nbasis] = lead
                nbasis += 1
                if src == 1:
                    keep[nkeep] = i
                    nkeep += 1
    return keep[:nkeep]


# ------------------------------------------------- 무작위 정보집합(Lee–Brickell p=2) 탐색
@nb.njit(cache=True)
def _search(G, L, n, max_trials, patience, seed):
    """span(G) 안에서 L과 반교환(L·v ≠ 0)하는 최소 무게 벡터의 상한을 찾는다."""
    np.random.seed(seed)
    kd, W = G.shape
    kl = L.shape[0]
    KW = (kl + 63) // 64
    best = n + 1
    since = 0
    R = np.empty_like(G)
    wt = np.empty(kd, dtype=np.int64)
    syn = np.empty((kd, KW), dtype=np.uint64)
    for trial in range(max_trials):
        R[:, :] = G
        perm = np.random.permutation(n)
        r = 0
        for c in perm:
            if r == kd:
                break
            w_, b_ = c >> 6, np.uint64(c & 63)
            p = -1
            for i in range(r, kd):
                if (R[i, w_] >> b_) & np.uint64(1):
                    p = i
                    break
            if p < 0:
                continue
            if p != r:
                for w in range(W):
                    t = R[r, w]; R[r, w] = R[p, w]; R[p, w] = t
            for i in range(kd):
                if i != r and (R[i, w_] >> b_) & np.uint64(1):
                    for w in range(W):
                        R[i, w] ^= R[r, w]
            r += 1
        improved = False
        for i in range(r):
            s = 0
            for w in range(W):
                s += _popcount(R[i, w])
            wt[i] = s
            for q in range(KW):
                syn[i, q] = 0
            for t in range(kl):
                par = 0
                for w in range(W):
                    par += _popcount(R[i, w] & L[t, w])
                if par & 1:
                    syn[i, t >> 6] |= np.uint64(1) << np.uint64(t & 63)
            nz = False
            for q in range(KW):
                if syn[i, q] != 0:
                    nz = True
            if nz and s < best:
                best = s; improved = True
        for i in range(r):
            for j in range(i + 1, r):
                nz = False
                for q in range(KW):
                    if syn[i, q] != syn[j, q]:
                        nz = True
                        break
                if not nz:
                    continue
                s = 0
                for w in range(W):
                    s += _popcount(R[i, w] ^ R[j, w])
                    if s >= best:
                        break
                if s < best:
                    best = s; improved = True
        since = 0 if improved else since + 1
        if best <= n and since >= patience:
            break
    return best


def code_parameters(ell, m, A, B, max_trials=2000, patience=100, seed=0):
    """CSS BB 코드의 (n, k, d_upper). k == 0이면 d = 0."""
    HX, HZ = bb_check_matrices(ell, m, A, B)
    n = HX.shape[1]
    NX, rX = gf2_nullspace(HX)
    NZ, rZ = gf2_nullspace(HZ)
    k = n - rX - rZ
    if k == 0:
        return n, 0, 0
    RX, _ = gf2_rref(HX)
    RZ, _ = gf2_rref(HZ)
    # X형 논리 연산자: ker(H_Z) \ rowspace(H_X),  Z형: ker(H_X) \ rowspace(H_Z)
    LX = NZ[_independent_rows(pack(RX), pack(NZ))]
    LZ = NX[_independent_rows(pack(RZ), pack(NX))]
    assert len(LX) == k and len(LZ) == k
    dZ = _search(pack(NX), pack(LX), n, max_trials, patience, seed)
    dX = _search(pack(NZ), pack(LZ), n, max_trials, patience, seed + 1)
    return n, k, int(min(dX, dZ))


def code_distance(ell, m, A, B, max_trials=2000, patience=100, seed=0):
    """joblib 워커용: 휴리스틱 거리 상한만 반환."""
    return code_parameters(ell, m, A, B, max_trials, patience, seed)[2]
