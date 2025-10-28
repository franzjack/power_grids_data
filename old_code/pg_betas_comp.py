# -*- coding: utf-8 -*-
"""
Numba-safe version (pure numeric kernels JIT-compiled; orchestration in Python)
"""

import numpy as np
from numba import njit
from tqdm import tqdm  # use only in Python, never inside @njit code

# ===============================
# JIT KERNELS (NUMERIC ONLY)
# ===============================

@njit
def byn_coef(N: np.int64) -> np.int64:
    # C(N,2) with integer arithmetic
    return (N * (N - 1)) // 2 if N >= 2 else 0

@njit
def sigmoid(th: float) -> float:
    # stable logistic
    # equivalent to np.exp(th)/(1+np.exp(th)), but numerically nicer
    if th >= 0.0:
        z = np.exp(-th)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(th)
        return z / (1.0 + z)

@njit
def freederivative2(theta: float, N_der: np.int64) -> float:
    return N_der * sigmoid(theta)

@njit
def freederivative2_vec(thetas: np.ndarray, quants: np.ndarray, out: np.ndarray) -> None:
    # thetas: (K,) float64; quants: (K,) int64; out: (K,) float64
    for i in range(thetas.size):
        out[i] = freederivative2(thetas[i], np.int64(quants[i]))

@njit
def linktype(a: np.int64, b: np.int64) -> np.int64:
    # preserves your original rule; ensure integer ops
    s = a + b
    if s < 5:
        return a * b - 1
    else:
        return s - 1

@njit
def reversetype_flat(lt: np.int64, countlist: np.ndarray):
    """
    Returns (l1_start, l1_end, l2_start, l2_end) as ints instead of nested tuples.
    countlist is expected to be length-3 int64 array: [count0, count1, count2]
    """
    c0, c1, c2 = np.int64(countlist[0]), np.int64(countlist[1]), np.int64(countlist[2])
    if lt == 0:
        return 0, c0, 0, c0
    if lt == 1:
        return 0, c0, c0, c0 + c1
    if lt == 2:
        return 0, c0, c0 + c1, c0 + c1 + c2
    if lt == 3:
        return c0, c0 + c1, c0, c0 + c1
    if lt == 4:
        return c0, c0 + c1, c0 + c1, c0 + c1 + c2
    # lt == 5
    return c0 + c1, c0 + c1 + c2, c0 + c1, c0 + c1 + c2

@njit
def Dderivative(idx: np.int64, ordlist: np.ndarray, thetE: np.ndarray, thetD: np.ndarray) -> float:
    val = 0.0
    n = ordlist.size
    for j in range(n):
        lt = linktype(np.int64(ordlist[idx]), np.int64(ordlist[j]))
        val += sigmoid(thetD[idx] + thetD[j] + thetE[lt])
    # subtract the self-term once (j = idx)
    lt_self = linktype(np.int64(ordlist[idx]), np.int64(ordlist[idx]))
    val -= sigmoid(thetD[idx] + thetD[idx] + thetE[lt_self])
    return val

@njit
def Ederivative(bt: np.int64, countlist: np.ndarray, thetE: np.ndarray, thetD: np.ndarray) -> float:
    val = 0.0
    l1s, l1e, l2s, l2e = reversetype_flat(bt, countlist)
    if l1s != l2s or l1e != l2e:
        # different blocks -> full rectangle
        for n in range(l1s, l1e):
            for m in range(l2s, l2e):
                val += sigmoid(thetD[n] + thetD[m] + thetE[bt])
    else:
        # same block -> upper triangle including diagonal
        for n in range(l1s, l1e):
            for m in range(n, l2e):
                val += sigmoid(thetD[n] + thetD[m] + thetE[bt])
    return val

@njit
def Dderivative2(idx: np.int64, thetD: np.ndarray) -> float:
    val = 0.0
    n = thetD.size
    for j in range(n):
        val += sigmoid(thetD[idx] + thetD[j])
    # subtract self once
    val -= sigmoid(thetD[idx] + thetD[idx])
    return val

@njit
def tweak_params(exval: float, theta: float, N_der: np.int64) -> float:
    # single-parameter step using freederivative2
    if exval - freederivative2(theta, N_der) < 0.0:
        theta -= 0.025
    else:
        theta += 0.025
    return theta

@njit
def tweak_paramsD(dval: float, idx: np.int64, ordlist: np.ndarray, thetE: np.ndarray, thetD: np.ndarray) -> None:
    if dval - Dderivative(idx, ordlist, thetE, thetD) < 0.0:
        thetD[idx] -= 0.025
    else:
        thetD[idx] += 0.025

@njit
def tweak_paramsE(Eval: float, bt: np.int64, countlist: np.ndarray, thetE: np.ndarray, thetD: np.ndarray) -> None:
    if Eval - Ederivative(bt, countlist, thetE, thetD) < 0.0:
        thetE[bt] -= 0.025
    else:
        thetE[bt] += 0.025

@njit
def tweak_paramsD2(dval: float, idx: np.int64, thetD: np.ndarray) -> None:
    if dval - Dderivative2(idx, thetD) < 0.0:
        thetD[idx] -= 0.025
    else:
        thetD[idx] += 0.025


# ===============================
# PYTHON ORCHESTRATION (SAFE TO USE tqdm/printing/etc.)
# ===============================

def check_condition(realvals: np.ndarray, thetas: np.ndarray, quants: np.ndarray, atol: float = 0.2) -> bool:
    buf = np.empty_like(thetas, dtype=np.float64)
    freederivative2_vec(thetas, quants.astype(np.int64, copy=False), buf)
    return np.allclose(buf, realvals, atol=atol)

def greedsearch_param2(realvals, NG, NL, NI, maxiter=30000, startguess=None, atol=0.2):
    """
    Greedy search for thetas with single-parameter updates.
    realvals: shape (6,) expected
    """
    realvals = np.asarray(realvals, dtype=np.float64)
    if startguess is None:
        startguess = np.zeros_like(realvals, dtype=np.float64)
    else:
        startguess = np.asarray(startguess, dtype=np.float64)

    # combinatorial counts (int64 to avoid overflow in byn_coef)
    NG = np.int64(NG); NL = np.int64(NL); NI = np.int64(NI)
    NGG = np.int64(byn_coef(NG))
    NLL = np.int64(byn_coef(NL))
    NII = np.int64(byn_coef(NI))
    NGL = np.int64(NG * NL)
    NGI = np.int64(NG * NI)
    NLI = np.int64(NL * NI)

    quants = np.array([NGG, NGL, NGI, NLL, NLI, NII], dtype=np.int64)
    thetas = startguess.copy()

    for _ in tqdm(range(maxiter), desc="greedsearch_param2"):
        for j in range(realvals.size):
            thetas[j] = tweak_params(realvals[j], thetas[j], quants[j])
        if check_condition(realvals, thetas, quants, atol=atol):
            break
    return thetas

def greedsearch_paramDD(realvalsD, realvalsE, ordlist, countlist, maxiter=300000):
    """
    ordlist: (N,) int64 node types
    countlist: (3,) int64 counts per type
    """
    realvalsD = np.asarray(realvalsD, dtype=np.float64)
    realvalsE = np.asarray(realvalsE, dtype=np.float64)
    ordlist = np.asarray(ordlist, dtype=np.int64)
    countlist = np.asarray(countlist, dtype=np.int64)

    thetasD = np.zeros(realvalsD.size, dtype=np.float64)
    thetasE = np.zeros(realvalsE.size, dtype=np.float64)

    for _ in tqdm(range(maxiter), desc="greedsearch_paramDD"):
        for k in range(realvalsD.size):
            tweak_paramsD(realvalsD[k], np.int64(k), ordlist, thetasE, thetasD)
        for z in range(realvalsE.size):
            tweak_paramsE(realvalsE[z], np.int64(z), countlist, thetasE, thetasD)

    # collect derivatives at the end
    dlist = [Dderivative(np.int64(k), ordlist, thetasE, thetasD) for k in range(realvalsD.size)]
    elist = [Ederivative(np.int64(z), countlist, thetasE, thetasD) for z in range(realvalsE.size)]
    return thetasD, thetasE, dlist, elist

def greedsearch_paramDD_gen(realvalsD, realvalsE, ordlist, countlist, maxiter=300000):
    """
    Variant that only iterates k in range(countlist[0]) for D updates (as in your code).
    """
    realvalsD = np.asarray(realvalsD, dtype=np.float64)
    realvalsE = np.asarray(realvalsE, dtype=np.float64)
    ordlist = np.asarray(ordlist, dtype=np.int64)
    countlist = np.asarray(countlist, dtype=np.int64)

    thetasD = np.zeros(realvalsD.size, dtype=np.float64)
    thetasE = np.zeros(realvalsE.size, dtype=np.float64)

    kmax = int(countlist[0])
    for _ in tqdm(range(maxiter), desc="greedsearch_paramDD_gen"):
        for k in range(kmax):
            tweak_paramsD(realvalsD[k], np.int64(k), ordlist, thetasE, thetasD)
        for z in range(realvalsE.size):
            tweak_paramsE(realvalsE[z], np.int64(z), countlist, thetasE, thetasD)

    dlist = [Dderivative(np.int64(k), ordlist, thetasE, thetasD) for k in range(kmax)]
    elist = [Ederivative(np.int64(z), countlist, thetasE, thetasD) for z in range(realvalsE.size)]
    return thetasD[:kmax], thetasE, dlist, elist

def greedsearch_paramDD2(realvalsD, maxiter=300000):
    realvalsD = np.asarray(realvalsD, dtype=np.float64)
    thetasD = np.zeros(realvalsD.size, dtype=np.float64)

    for _ in tqdm(range(maxiter), desc="greedsearch_paramDD2"):
        for k in range(realvalsD.size):
            tweak_paramsD2(realvalsD[k], np.int64(k), thetasD)

    dlist = [Dderivative2(np.int64(k), thetasD) for k in range(realvalsD.size)]
    return thetasD, dlist
