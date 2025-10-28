# -*- coding: utf-8 -*-
"""
Numba-safe refactor of observables & fast update utilities.
- All numeric kernels use @njit(nopython=True).
- Removed copy/deepcopy; use NumPy copies where needed.
- Fixed divisions (use // for combinatorial integers; / for floats).
- Safer triangle-update logic and slicing.
"""

import numpy as np
from numba import njit
import old_code.pg_betas_comp as bc  # used only in comp_obs_and_betas (Python)

# -------- utilities ---------------

@njit
def compute_k_triangle(mtx):
    """
    Counts:
      tricount  = total '2-paths closed' over edges (used to get triangles)
      tricount2 = sum over edges of C(localtricount, 2)  (k-triangle related)
    Returns:
      (triangles, tricount2//2) as ints.
    """
    n = mtx.shape[0]
    tricount = 0
    tricount2 = 0
    for i in range(n):
        for j in range(n):
            if mtx[i, j] == 1:
                localtricount = 0
                for k in range(n):
                    if mtx[i, k] == 1:
                        localtricount += mtx[k, j]
                if localtricount >= 2:
                    tricount2 += (localtricount * (localtricount - 1)) // 2
                tricount += localtricount
    return (tricount // 6, tricount2 // 2)

@njit
def compute_2_triangle(mtx):
    """
    Returns tricount2//2 only (see compute_k_triangle).
    """
    n = mtx.shape[0]
    tricount2 = 0
    for i in range(n):
        for j in range(n):
            if mtx[i, j] == 1:
                localtricount = 0
                for k in range(n):
                    if mtx[i, k] == 1:
                        localtricount += mtx[k, j]
                if localtricount >= 2:
                    tricount2 += (localtricount * (localtricount - 1)) // 2
    return tricount2 // 2

@njit
def linktype(a, b):
    s = a + b
    if s < 5:
        return a * b - 1
    else:
        return s - 1

@njit
def deg_distr(mtx):
    # degree per node; returns float64 for generality
    return mtx.sum(axis=1).astype(np.float64)

@njit
def check_existing_triangle(i, k, mtx):
    """
    Count number of nodes l such that (k,l) and (l,i) are edges.
    """
    n = mtx.shape[0]
    cnt = 0
    for l in range(n):
        if mtx[k, l] == 1:
            cnt += mtx[l, i]
    return cnt

@njit
def change_triang3(mtx, i, j):
    """
    For a proposed flip on edge (i,j):
      - count = number of common neighbors between i and j (triangles touching edge)
      - kcount + lcount + rcount is the sum of 'k-triangle' contributions affected
    Returns:
      (count, total_k_contrib)
    """
    n = mtx.shape[0]
    count = 0
    kcount = 0  # C(count,2) when count>=2
    lcount = 0
    rcount = 0

    for k in range(n):
        if mtx[i, k] == 1 and mtx[k, j] == 1:
            count += 1
            # subtract direct edge mtx[i,j] from triangle checks
            lcount += check_existing_triangle(i, k, mtx) - mtx[i, j]
            rcount += check_existing_triangle(j, k, mtx) - mtx[i, j]

    if count >= 2:
        kcount = (count * (count - 1)) // 2

    return count, (kcount + lcount + rcount)

# ---------- obs computation methods ---------------

@njit
def obs_edg_tri_2tri(mtx, q1, q2, q3):
    e_gg = mtx[0:q1, 0:q1].sum() / 2.0
    e_ll = mtx[q1:(q1+q2), q1:(q1+q2)].sum() / 2.0
    e_ii = mtx[(q1+q2):, (q1+q2):].sum() / 2.0
    e_gl = mtx[0:q1, q1:(q1+q2)].sum() * 1.0
    e_gi = mtx[0:q1, (q1+q2):].sum() * 1.0
    e_li = mtx[q1:(q1+q2), (q1+q2):].sum() * 1.0
    t, tk = compute_k_triangle(mtx)
    out = np.empty(8, dtype=np.float64)
    out[0] = e_gg; out[1] = e_gl; out[2] = e_gi
    out[3] = e_ll; out[4] = e_li; out[5] = e_ii
    out[6] = float(t); out[7] = float(tk)
    return out

def obs_er_tri_2tri(mtx):
    e = mtx.sum() / 2.0
    t, tk = compute_k_triangle.py_func(mtx)  # call njit impl from Python
    return np.array([e, float(t), float(tk)], dtype=np.float64)

@njit
def obs_edg(mtx, q1, q2, q3):
    e_gg = mtx[0:q1, 0:q1].sum() / 2.0
    e_ll = mtx[q1:(q1+q2), q1:(q1+q2)].sum() / 2.0
    e_ii = mtx[(q1+q2):, (q1+q2):].sum() / 2.0
    e_gl = mtx[0:q1, q1:(q1+q2)].sum() * 1.0
    e_gi = mtx[0:q1, (q1+q2):].sum() * 1.0
    e_li = mtx[q1:(q1+q2), (q1+q2):].sum() * 1.0
    out = np.empty(6, dtype=np.float64)
    out[0] = e_gg; out[1] = e_gl; out[2] = e_gi
    out[3] = e_ll; out[4] = e_li; out[5] = e_ii
    return out

@njit
def obs_avgdeg(mtx, q1, q2, q3):
    # average degree per block (counts edges by row sums)
    d1 = mtx[0:q1, :].sum() / max(1, q1)
    d2 = mtx[q1:(q1+q2), :].sum() / max(1, q2)
    d3 = mtx[(q1+q2):, :].sum() / max(1, q3)
    return d1, d2, d3

@njit
def obs_edg_ddeg(mtx, q1, q2, q3):
    obsD = deg_distr(mtx)
    obsE = obs_edg(mtx, q1, q2, q3)
    out = np.empty(obsD.size + obsE.size, dtype=np.float64)
    out[:obsD.size] = obsD
    out[obsD.size:] = obsE
    return out

@njit
def obs_edg_dgen(mtx, q1, q2, q3):
    obsD = deg_distr(mtx)
    obsE = obs_edg(mtx, q1, q2, q3)
    out = np.empty(q1 + obsE.size, dtype=np.float64)
    out[:q1] = obsD[:q1]
    out[q1:] = obsE
    return out

@njit
def avg_degreetype(mx, bustypes):
    n = mx.shape[0]
    dgen = 0.0; dload = 0.0; dint = 0.0
    cgen = 0; cload = 0; cint = 0
    for i in range(n):
        di = mx[i, :].sum()
        t = bustypes[i]
        if t == 1:
            dgen += di; cgen += 1
        elif t == 2:
            dload += di; cload += 1
        else:
            dint += di; cint += 1
    if cgen == 0: cgen = 1
    if cload == 0: cload = 1
    if cint == 0: cint = 1
    return dgen / cgen, dload / cload, dint / cint, cgen, cload, cint

# ------------- fast_obs methods (Numba-safe) -------------

@njit
def fast_obsDD(obs, move, i, j):
    """
    Degree-only fast update: move in {0,1} is new value at (i,j).
    delta = +1 if edge added, -1 if removed.
    """
    dmove = 2 * move - 1  # 1-> +1, 0-> -1
    newobs = obs.copy()
    newobs[i] += dmove
    newobs[j] += dmove
    return newobs

@njit
def fast_obs_er_tri_2tri(past_obs, mtx, move, i, j):
    """
    past_obs = [e, t, tk]; updates after flipping (i,j).
    """
    newobs = past_obs.copy()
    t, kcount = change_triang3(mtx, i, j)
    if move == 1:
        newobs[0] += 1.0
        newobs[1] += float(t)
        newobs[2] += float(kcount)
    else:
        newobs[0] -= 1.0
        newobs[1] -= float(t)
        newobs[2] -= float(kcount)
    return newobs

@njit
def fast_obs_edg(past_obs, mtx, ordlist, move, i, j):
    newobs = past_obs.copy()
    addtype = linktype(ordlist[i], ordlist[j])
    if move == 1:
        newobs[addtype] += 1.0
    else:
        newobs[addtype] -= 1.0
    return newobs

@njit
def fast_obs_edg_tri_2tri(past_obs, mtx, ordlist, move, i, j):
    """
    past_obs = [e_gg,e_gl,e_gi,e_ll,e_li,e_ii,t,tk]
    """
    newobs = past_obs.copy()
    t, kcount = change_triang3(mtx, i, j)
    addtype = linktype(ordlist[i], ordlist[j])
    if move == 1:
        newobs[addtype] += 1.0
        newobs[6] += float(t)
        newobs[7] += float(kcount)
    else:
        newobs[addtype] -= 1.0
        newobs[6] -= float(t)
        newobs[7] -= float(kcount)
    return newobs

@njit
def alt_fast_obs_edg_tri_2tri(past_obs, mtx, ordlist, move, i, j):
    """
    Alternative where Δ on index 6 uses k = 3*t - kcount (as in your code).
    """
    newobs = past_obs.copy()
    t, kcount = change_triang3(mtx, i, j)
    k = 3 * t - kcount
    addtype = linktype(ordlist[i], ordlist[j])
    if move == 1:
        newobs[addtype] += 1.0
        newobs[6] += float(k)
    else:
        newobs[addtype] -= 1.0
        newobs[6] -= float(k)
    return newobs

@njit
def fast_obs_ddeg(past_obs, mtx, ordlist, move, i, j):
    # recompute full degree distribution (simple & safe)
    return deg_distr(mtx)

@njit
def fast_obs_edg_ddeg(past_obs, mtx, ordlist, move, i, j):
    # past_obs = [deg (N entries), six edge-type counts]
    N = mtx.shape[0]
    newobsE = fast_obs_edg(past_obs[N:], mtx, ordlist, move, i, j)
    newobsD = deg_distr(mtx)
    out = np.empty(N + newobsE.size, dtype=np.float64)
    out[:N] = newobsD
    out[N:] = newobsE
    return out

@njit
def _count_value(ordlist, value):
    c = 0
    for i in range(ordlist.size):
        if ordlist[i] == value:
            c += 1
    return c

@njit
def fast_obs_edg_dgen(past_obs, mtx, ordlist, move, i, j):
    """
    past_obs = [deg[:q1], six edge-type counts]
    We infer q1 as count of '1' in ordlist to avoid Python-side info.
    """
    q1 = _count_value(ordlist, 1)
    newobsE = fast_obs_edg(past_obs[q1:], mtx, ordlist, move, i, j)
    fullD = deg_distr(mtx)
    out = np.empty(q1 + newobsE.size, dtype=np.float64)
    out[:q1] = fullD[:q1]
    out[q1:] = newobsE
    return out

# ---------- model-level glue (Python) ----------

def comp_obs_and_betas(modtype, ordmat, ordlist, countlist, maxiter=7000, startguess=np.array([1, -0.2])):
    """
    Wrapper that computes observables and calls parameter estimators in bc.
    Note: bc.* functions are assumed to be Python/Numpy/Numba orchestrations.
    """
    q1, q2, q3 = int(countlist[0]), int(countlist[1]), int(countlist[2])

    if modtype == '_edg_tri_2tri':
        obs = obs_edg_tri_2tri(ordmat, q1, q2, q3)
        kbetas = bc.greedsearch_param2(obs[:6], q1, q2, q3, maxiter=maxiter)
        betas = np.append(kbetas * 1.1, startguess)
        return obs, betas

    if modtype == '_edg_ddeg':
        kedgobs_T = obs_edg(ordmat, q1, q2, q3)
        realvals = deg_distr(ordmat)
        obs = np.concatenate((realvals, kedgobs_T))
        thetasD, thetasE, dlist, elist = bc.greedsearch_paramDD(realvals, kedgobs_T, ordmat, ordlist, countlist, maxiter=300000)
        betas = np.concatenate((thetasD, thetasE))
        return obs, betas

    if modtype == '_edg':
        obs = obs_edg(ordmat, q1, q2, q3)
        betas = bc.greedsearch_param2(obs, q1, q2, q3, maxiter=maxiter)
        return obs, betas

    if modtype == '_edg_dgen':
        kedgobs_T = obs_edg(ordmat, q1, q2, q3)
        realvals = deg_distr(ordmat)
        thetasD, thetasE, dlist, elist = bc.greedsearch_paramDD_gen(realvals, kedgobs_T, ordmat, ordlist, countlist, maxiter=300000)
        betas = np.concatenate((thetasD[:q1], thetasE))
        obs = np.concatenate((realvals[:q1], kedgobs_T))
        return obs, betas

    raise ValueError(f"Unknown modtype: {modtype}")
