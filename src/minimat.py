from time import time
from itertools import combinations
from multiprocessing import Pool
from functools import partial
import multiprocessing

from src.hypergraph import *
from src.comparison import *
from src.datamat import *

def inf_cyclic(CF1, CF2, d):
    """
    Test whether CF1 <= CF2 or not
    where Mi is the matroid defined by CFi, with ground set [d]
    """
    return inf_hyper(cyclic_to_hyper(CF1, d), cyclic_to_hyper(CF2, d))

def solve_int(XT, r):
    """
    Input: labeled hypergraph XT (2-list of sets); r (integer)
    Output: * flag (bool) indicating if modifications have been done
            * HTbis (2-list of sets) fixed hypegraph

    Fix all submodularity failures that require no choice
    depending on the situation given by r
    (no rk(cyclic flat) < r included)
    """
    flag = False
    T2, T3 = XT[0].copy(), XT[1].copy()
    # Fix Case 4
    for i in range(len(T2)):
        for j in range(i+1,len(T2)):
            if len(T2[i] & T2[j]) == 1 and not is_inside(T2[i] | T2[j], XT[1]):
                flag = True
                T3.append(T2[i] | T2[j])

    T2, T3 = rem_sub([T2, T3])

    if r >= 3:
        # Fix Case 2
        for i in range(len(T3)):
            for j in range(len(T2)):
                if len(T3[i] & T2[j]) > 1 and not T2[j] <= T3[i]:
                    flag = True
                    T3 = [ T3[k] for k in range(len(T3)) if k != i ] + [T3[i] | T2[j]]
        T2, T3 = rem_sub([T2, T3])

    if r == 3:
        # Fix Case 3
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) > 1:
                    flag = True
                    T2 = [ T2[k] for k in range(len(T2)) if k not in [i,j] ] + [T2[i] | T2[j]]
        T2, T3 = rem_sub([T2, T3])

    if r >= 4:
        # Fix Case 1
        for i in range(len(T3)):
            for j in range(i+1,len(T3)):
                S = T3[i] & T3[j]
                if len(S) > 2 and not is_inside(S, T2):
                    flag = True
                    T3 = [ T3[k] for k in range(len(T3)) if k not in [i,j] ] + [T3[i] | T3[j]]
        T2, T3 = rem_sub([T2, T3])

    return flag, [T2, T3]

def detect_mat_case(XT, c, r):
    """
    Input: labeled hypergraph XT (2-list of sets), c (int)
    Output: c (int), (i,j) (int, int)

    Perform the test of the case 'c' of submodularity fail
    Return indices (i,j) of corresponding 'c'-failure, if found
    If no failure, return '0, ()' (the hypergraph defines a matroid)
    """

    T2, T3 = XT
    if c == 1:
        for i in range(len(T3)):
            for j in range(i+1,len(T3)):
                S = T3[i] & T3[j]
                if len(S) > 2 and not is_inside(S, T2):
                    return 1, (i,j)
    if c == 2:
        for i in range(len(T3)):
            for j in range(len(T2)):
                if len(T3[i] & T2[j]) > 1 and not T2[j] <= T3[i]:
                    return 2, (i,j)
    if c == 3 and r<=3:
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) > 1:
                    return 3, (i,j)

    if c == 4 and r<=3:
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) == 1 and not is_inside(T2[i] | T2[j], T3):
                    return 4, (i,j)
    return 0, () # No failure found

def detect_mat(XT, r):
    """
    Input: labeled hypergraph XT (2-list of sets)
    Perform submodularity fail tests
    in a chosen case order
    """
    seq = [3, 2, 1, 4] # Custom case detection order (heuristic)
    for c in seq:
        case, ind = detect_mat_case(XT, c, r)
        if case != 0:
            return case, ind
    return 0, ()

def comp_leaves(HT, r, pproc=False):
    """
    Input: labeled hypergraph XT (2-list of sets), r (int), pproc (Bool)
    Output: a list Lcand of labeled hypergraphs

    Compute all hypergraphs corresponding to the matroid extensions of
    the input hypergraph, and that contain no additional Type < r edge.

    Option pproc performs a call to solve_int before any recursive call.
    This does not change the number of candidates, see the doc of solve_int for further details.
    """
    # no  cyclic flat of rank < r included
    XT, P = HT
    XT = [XT[0].copy(), XT[1].copy()]
    Lcand = []

    # We first reduce the hypergraph as we added some edges
    XT = rem_sub(XT)
    if pproc:
        flag = True
        # Perform preprocessing as many times as needed
        while flag:
            flag, XT = solve_int(XT, r)
    c, ind = detect_mat(XT, r)

    # XT is already a matroid
    if c == 0:
      Lcand.append([XT, P])
    else:
        T2,  T3 = XT
        if c == 1:
            e1, e2 = T3[ind[0]], T3[ind[1]]
            if r <= 4:
                T3bis = [ T3[i] for i in range(len(T3)) if i not in ind ]
                T3bis.append(e1 | e2)
                Lcand += comp_leaves([[T2, T3bis], P], r, pproc=pproc)
            if r <= 3:
                T2bis = T2.copy() + [e1 & e2]
                Lcand += comp_leaves([[T2bis, T3], P], r, pproc=pproc)

        elif c == 2:
            e1, e2 = T3[ind[0]], T2[ind[1]]
            if r <= 4:
                T3bis = [ T3[i] for i in range(len(T3)) if i != ind[0] ]
                T3bis.append(e1 | e2)
                Lcand += comp_leaves([[T2, T3bis], P], r, pproc=pproc)
            if r <= 2:
                HTid = identify(HT, e1 & e2)
                Lcand += comp_leaves(HTid, r, pproc=pproc)

        elif c == 3:
            e1, e2 = T2[ind[0]], T2[ind[1]]
            if r <= 3:
                T2bis = [ T2[i] for i in range(len(T2)) if i not in ind ]
                T2bis.append(e1 | e2)
                Lcand += comp_leaves([[T2bis, T3], P], r, pproc=pproc)
            if r <= 2:
                HTid = identify(HT, e1 & e2)
                Lcand += comp_leaves(HTid, r, pproc=pproc)

        elif c == 4:
            e1, e2 = T2[ind[0]], T2[ind[1]]
            T3bis = T3.copy() + [e1 | e2]
            Lcand += comp_leaves([[T2, T3bis], P], r, pproc=pproc)

    return Lcand

def redund_edge(XT, x, i):
    # Test whether the edge x is redundant in XT
    if i <= 2:
        return False # No loop nor 2-pts
    if i == 3:
        return is_inside(x, XT[0])
    if i == 4:
        return is_inside(x, XT[1]) or inter_size(x, XT[0], 3)

def add_edge(HT, x, i):
    # Add a non-redundant edge x of Type(i-1) to HT
    # if i<=2, reduce the lab hypergraph
    if i == 1:
        return remove(HT, min(HT[1][i]))
    if i == 2:
        return identify(HT, x)
    if i == 3:
        return [ [HT[0][0]+[x], HT[0][1]] , HT[1]]
    if i == 4:
        return [ [HT[0][0], HT[0][1]+[x]] , HT[1]]

def minimal_extensions(CF, d, S=[1,2,3,4], v=0, preprocess=False, n_procs=1):
    # CF[i] are the cyclic flats of rk i of the input matroid M ##
    """
    Input: CF (4-list of sets), d (int), S (sublist of [1,2,3,4]), v(Int), preproccess (Bool)
    CF[i] are the cyclic flats of rk i of the input matroid M with ground set [d].

    Output: a list cumins of 4-list of sets

    The function computes all matroid exensions of M, that belong to all S_i, for i in the input list S.
    These extensions are encoded by their cyclic flats in the output (as for the input M).

    Integer v>0 controls verbosity level during computations and preprocess enable preprocessing in
    comp_leaves function.
    """
    assert set(S) <= {1,2,3,4}, "Wrong input for S"
    assert len(CF) == 4 and all(type(t)==list and len(t)==0 or type(t[0])==set for t in CF), "Wrong input for CF"
    assert d > 0 and all( s <= d for cf in CF for c in cf for s in c ), "Wrong input for d"

    if n_procs <= 0:
        n_procs = multiprocessing.cpu_count()

    HT = cyclic_to_hyper(CF, d)
    XT, P = HT
    tleaf, tmin1, tmin2 = 0, 0, 0

    v>0 and print("Compute minimals in " + ", ".join(["S{}(M)".format(i) for i in S]))

    Lcumins = []

    for i in sorted(S, reverse=True): # We start by highest Si, being smallest
        #print(i)
        Lcands = []
        for Lind in combinations(range(d), i): # Take a circuit of size i
            x = { min(P[ind]) for ind in Lind } # Its representative
            if not redund_edge(XT, x, i): # Check x is not redundant in XT
                t = time()
                Lcands.append(comp_leaves(add_edge(HT, x, i), i, pproc=preprocess) if i>1 else [add_edge(HT, x, 1)]) # compute extension matroids
                tleaf += time() - t
                v==2 and print("Add {}: {:.2g}s".format(str(x).replace(" ", ""),time()-t))
                #v>2 and print("Add {}: {:,} cands ({:.2g}s); {:,} S{}-current mins ({:.2g}s) ".format(str(x).replace(" ", ""),len(cands), t1-t, len(Lcumins[i-1]), i, time()-t1))
        t1 = time()
        with Pool(n_procs) as p:
            LX = list(p.map(partial(poset_mins_part, LX=Lcumins, f=inf_hyper), Lcands)) # select minimal ones
        tmin1 += time() - t1
        t2 = time()
        plop = parallel_min_poset(LX, inf_hyper, n_procs=n_procs)
        Lcumins.append([])
        for pl in plop:
            Lcumins[-1].extend(pl)
        tmin2 += time() - t2
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))


    v>0 and print("Time elapsed {:.2f}s (total) ; {:.2f}s (cand) ; {:.2f}s (mins1) ; {:.2f}s (mins2)\n".format(tleaf+tmin1+tmin2, tleaf, tmin1, tmin2))
    cumins = []
    for l in Lcumins:
        cumins += [ hyper_to_cyclic(ll,d) for ll in l ]
    return cumins