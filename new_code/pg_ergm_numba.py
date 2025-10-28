# -*- coding: utf-8 -*-
"""
Fully Numba-accelerated EEsparse (self-contained, no SciPy/tqdm).
- Provides: EEsparse_numba(...)
- Includes Numba-safe connectivity (BFS), proposal, and param update.
- Replace the @njit stubs obs_comp() and fast_obs() with your own JIT'ed versions.
"""

import numpy as np
from numba import njit
import time
from scipy.sparse import csr_matrix

# --------------------------
# Numeric helpers (Numba-safe)
# --------------------------

@njit
def ordered_buslist(q1: int, q2: int, q3: int):
    """Return ordlist of length q1+q2+q3 with values in {1,2,3}."""
    n = q1 + q2 + q3
    out = np.ones(n, dtype=np.int64)
    for i in range(q1, q1 + q2):
        out[i] = 2
    for i in range(q1 + q2, n):
        out[i] = 3
    return out

@njit
def dtype_ham(x: np.ndarray, params: np.ndarray) -> float:
    """Hamiltonian as dot for 1D arrays."""
    return float(np.dot(x, params))

@njit
def compute_p(prop: float, current: float) -> float:
    """MH acceptance probability (unnormalized, cap with cond outside)."""
    diff = prop - current
    if diff >= 0.0:
        return 1.0
    else:
        return float(np.exp(diff))

@njit
def random_pair(n: int) -> tuple:
    """Pick distinct (i, j) uniformly without recursion."""
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    while j == i:
        j = np.random.randint(0, n)
    return i, j

@njit
def flip_edge_sym_inplace(m: np.ndarray, i: int, j: int) -> int:
    """
    Flip undirected edge (i,j) in-place for adjacency matrix m in {0,1}.
    Returns the NEW value at (i,j) after the flip (0 or 1).
    """
    new_val = 1 - m[i, j]
    m[i, j] = new_val
    m[j, i] = new_val
    return new_val


@njit
def generate_connected_adj(size: np.int64) -> np.ndarray:
    """
    Simple ring graph adjacency (always connected).
    dtype=uint8 to save space, but bool/int8 are fine too.
    """
    mat = np.zeros((size, size), dtype=np.uint8)
    for i in range(size - 1):
        mat[i, i + 1] = 1
        mat[i + 1, i] = 1
    mat[size - 1, 0] = 1
    mat[0, size - 1] = 1
    return mat


def reorder_rows(a: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """
    Reorder rows/cols by a permutation vector (no deepcopy).
    """
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))
    newa = a[idx, :]
    newaa = newa[:, idx]
    return newaa

@njit
def is_connected(adj: np.ndarray) -> bool:
    """
    BFS-based connectivity check for an undirected simple graph (0/1 adj).
    Returns True iff the graph is connected.
    """
    n = adj.shape[0]
    if n == 0:
        return True
    visited = np.zeros(n, dtype=np.uint8)
    queue = np.empty(n, dtype=np.int64)
    qstart = 0
    qend = 0
    queue[qend] = 0
    qend += 1
    visited[0] = 1

    while qstart < qend:
        v = queue[qstart]
        qstart += 1
        row = adj[v]
        for u in range(n):
            if row[u] != 0 and visited[u] == 0:
                visited[u] = 1
                queue[qend] = u
                qend += 1

    for i in range(n):
        if visited[i] == 0:
            return False
    return True

@njit
def change_param(obs: np.ndarray, robs: np.ndarray, par: np.ndarray, a: float, c: float) -> np.ndarray:
    """
    Parameter update rule (Numba-safe). Magnitude-based step with floor c.
    npar[i] += a * max(|par[i]|, c) * (-sign(obs[i] - robs[i])).
    """
    npar = par.copy()
    for i in range(npar.size):
        mag = a * (abs(npar[i]) if abs(npar[i]) > c else c)
        # sign(x) for floats; returns -1, 0, or 1
        diff = obs[i] - robs[i]
        s = 0.0
        if diff > 0.0:
            s = 1.0
        elif diff < 0.0:
            s = -1.0
        npar[i] += -mag * s
    return npar


# --------------------------
# Fully Numba-accelerated EEsparse
# --------------------------


@njit
def EEsparse(startmtx: np.ndarray,
                   observables: np.ndarray,
                   params: np.ndarray,
                   countlist: np.ndarray,
                   obs_comp,
                   fast_obs,
                   maxiter: int,
                   alpha: float,
                   c: float,
                   n_step: int):
    """
    Numba-accelerated parameter estimation loop with connectivity constraint.
    Requirements:
      - startmtx: (N,N) uint8/int8/int32 adjacency matrix in {0,1}
      - observables: (K,) target statistics (float64)
      - params: (K,) initial parameters (float64)
      - countlist: (3,) node-type counts [q1,q2,q3] (int64)
      - maxiter, alpha, c, n_step: scalars
      - obs_comp, fast_obs: compiled above (replace with your own JIT'ed versions)
    Returns:
      EEparams: (K,) float64 (mean of last 10% parameter samples, or last params)
      obslist: (K, T_kept) float64 (stack of accepted observables, transposed)
      paramEE: (K, S) float64 (parameter trajectory, transposed)
    """
    n = startmtx.shape[0]
    q1, q2, q3 = int(countlist[0]), int(countlist[1]), int(countlist[2])

    # current state
    obs = obs_comp(startmtx, q1, q2, q3)
    nparam = params.copy()
    oldham = dtype_ham(obs, nparam)
    mtx = startmtx.copy()

    # prealloc buffers (upper bounds, then trimmed)
    # store at most maxiter entries (accepted obs and updated params)
    oblist = np.zeros((maxiter, observables.size), dtype=np.float64)
    paramEE = np.zeros((maxiter, nparam.size), dtype=np.float64)
    obs_ptr = 0
    param_ptr = 0

    # ordlist (if your fast_obs needs it)
    ordlist = ordered_buslist(q1, q2, q3)

    count = 0
    # bigcount kept conceptually (unused in JIT version, but preserved if needed)
    # bigcount = 1

    for _ in range(maxiter):
        cond = 1.0

        # propose: copy + flip one random edge
        nmtx = mtx.copy()
        i, j = random_pair(n)
        move = flip_edge_sym_inplace(nmtx, i, j)

        # fast observable update
        newobs = fast_obs(obs, nmtx, ordlist, move, i, j)
        newham = dtype_ham(newobs, nparam)
        p = compute_p(newham, oldham)

        # Connectivity check only when REMOVING an edge (move == 0)
        if move == 0:
            if not is_connected(nmtx):
                cond = 0.0

        # MH accept
        if np.random.random() < p * cond:
            mtx = nmtx
            oldham = newham
            obs = newobs
            count += 1

            # record accepted observables
            if obs_ptr < oblist.shape[0]:
                oblist[obs_ptr, :] = obs
                obs_ptr += 1

            # periodically update parameters
            if count == n_step:
                count = 0
                # bigcount += 1
                nparam = change_param(obs, observables, nparam, alpha, c)
                if param_ptr < paramEE.shape[0]:
                    paramEE[param_ptr, :] = nparam
                    param_ptr += 1
        # else: reject and continue

    # Trim buffers to actual sizes
    paramEE = paramEE[:param_ptr, :]
    oblist = oblist[:obs_ptr, :]

    # Average last 10% of parameter samples, or return last params if empty
    if param_ptr > 0:
        tail_start = int(0.9 * param_ptr)
        if tail_start < param_ptr:
            tail = paramEE[tail_start:]
            EEparams = np.zeros(tail.shape[1], dtype=np.float64)
            for j in range(tail.shape[1]):
                s = 0.0
                for i in range(tail.shape[0]):
                    s += tail[i, j]
                EEparams[j] = s / max(1, tail.shape[0])
        else:
            EEparams = paramEE[-1]
    else:
        EEparams = nparam

    obslist = oblist.T
    return EEparams, obslist, paramEE.T


@njit
def eesparse_step(mtx, obs, params, observables, alpha, c, count, n_step, ordlist, countlist, obs_comp, fast_obs):
    """
    Perform one iteration of EEsparse MCMC.
    Returns updated (mtx, obs, params, count).
    """
    n = mtx.shape[0]
    i, j = random_pair(n)
    move = flip_edge_sym_inplace(mtx, i, j)

    new_obs = fast_obs(obs, mtx, ordlist, move, i, j)
    new_ham = dtype_ham(new_obs, params)
    old_ham = dtype_ham(obs, params)
    p = compute_p(new_ham, old_ham)

    cond = 1.0
    if move == 0:
        # check connectivity only on edge removal
        if not is_connected(mtx):
            cond = 0.0

    if np.random.random() < p * cond:
        # accept
        obs = new_obs
        count += 1
        if count == n_step:
            count = 0
            params = change_param(obs, observables, params, alpha, c)
    else:
        # reject → revert edge
        flip_edge_sym_inplace(mtx, i, j)

    return mtx, obs, params, count


# ======================================================
# === Outer Python driver with progress display ========
# ======================================================

def EEsparse_with_timer(startmtx, observables, params, countlist, obs_comp, fast_obs,
                        maxiter=10000, alpha=0.05, c=0.01, n_step=10,
                        print_every=1000):
    """
    EEsparse MCMC with live ETA and progress display using time.time().
    """
    mtx = startmtx.copy()
    obs = obs_comp(mtx, countlist[0], countlist[1], countlist[2])
    nparam = params.copy()
    ordlist = ordered_buslist(countlist[0], countlist[1], countlist[2])
    count = 0
    paramEE = np.zeros(((maxiter//print_every)-1, nparam.size), dtype=np.float64)
    obslist = np.zeros(((maxiter//print_every)-1, obs.size), dtype=np.float64)

    start_time = time.time()
    j=0
    for i in range(maxiter):
        mtx, obs, nparam, count = eesparse_step(
            mtx, obs, nparam, observables, alpha, c, count, n_step, ordlist, countlist, obs_comp, fast_obs
        )

        # Live progress updates (every print_every iterations)
        if i % print_every == 0 and i > 0:
            elapsed = time.time() - start_time
            speed = i / max(elapsed, 1e-9)
            eta = (maxiter - i) / max(speed, 1e-9)
            bar_len = 30
            filled = int(bar_len * i / maxiter)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {i/maxiter:.1%} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                end="",
            )
            paramEE[j, :] = nparam
            obslist[j, :] = obs
            j += 1

    total_time = time.time() - start_time
    print(f"\nCompleted {maxiter} iterations in {total_time:.2f}s")

    return nparam, obslist, paramEE.T



@njit
def mh_step(mtx, obs, oldham, params, ordlist, countlist, obs_comp, fast_obs):
    n = mtx.shape[0]
    i, j = random_pair(n)
    move = flip_edge_sym_inplace(mtx, i, j)
    new_obs = fast_obs(obs, mtx, ordlist, move, i, j)
    new_ham = dtype_ham(new_obs, params)
    p = compute_p(new_ham, oldham)

    cond = 1.0
    if move == 0:  # only check connectivity on edge removal
        if not is_connected(mtx):
            cond = 0.0

    accepted = False
    if np.random.random() < p * cond:
        obs = new_obs
        oldham = new_ham
        accepted = True
    else:
        # reject, revert flip
        flip_edge_sym_inplace(mtx, i, j)

    return mtx, obs, oldham, accepted


@njit
def mh_batch(mtx, obs, oldham, params, ordlist, countlist, batch_size, obs_comp, fast_obs):
    accepted = 0
    for _ in range(batch_size):
        mtx, obs, oldham, acc = mh_step(mtx, obs, oldham, params, ordlist, countlist, obs_comp, fast_obs)
        if acc:
            accepted += 1
    return mtx, obs, oldham, accepted


# =====================================================
# === Python wrapper with timer-based progress ========
# =====================================================

def pg_MHergm_conn_timer(startmtx, observables, params, countlist, obs_comp, fast_obs,
                         maxiter=10000, batch_size=100, print_every=10, save_every = 10):
    """
    MCMC sampler for connected ERGMs (no parameter updates).
    - Uses time.time() for live ETA and progress.
    - Batches MH steps to reduce Python→Numba overhead.
    """
    mtx = startmtx.copy()
    ordlist = ordered_buslist(countlist[0], countlist[1], countlist[2])
    obs = obs_comp(mtx, countlist[0], countlist[1], countlist[2])
    oldham = dtype_ham(obs, params)
    synth = []
    oblist = []

    start_time = time.time()
    total_accepted = 0
    n_batches = maxiter // batch_size

    for b in range(n_batches):
        mtx, obs, oldham, acc = mh_batch(
            mtx, obs, oldham, params, ordlist, countlist, batch_size, obs_comp, fast_obs
        )
        total_accepted += acc
        if b > n_batches // 1.3:
            if b % save_every == 0:
                oblist.append(obs.copy())
                synth.append(csr_matrix(mtx.copy()))

        # === progress printing ===
        if b % print_every == 0 and b > 0:
            elapsed = time.time() - start_time
            completed = (b + 1) * batch_size
            speed = completed / max(elapsed, 1e-9)
            eta = (maxiter - completed) / max(speed, 1e-9)
            bar_len = 30
            filled = int(bar_len * completed / maxiter)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"\r[{bar}] {completed/maxiter:.1%} | Accepted: {total_accepted} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                end=""
            )

    print(f"\nCompleted {maxiter} MH iterations in {time.time() - start_time:.2f}s")
    print(f"Total accepted moves: {total_accepted}")

    obslist = np.array(oblist[int(len(oblist)/1.3):]).T
    mean_list = [np.mean(ob) for ob in obslist]
    return mean_list, synth, obslist



