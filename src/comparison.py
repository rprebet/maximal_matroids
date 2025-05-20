from src.hypergraph import supp, inf_subs, replace, remove, is_inside, diff_size
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
from itertools import repeat
from functools import partial
from math import log2

def test_T3(l, X2):
    if is_inside(l, X2[1]):
        return True
    if diff_size(l, X2[0], 1):
        return True
    return False

def inf_hyper(H1, H2):
    """
    Test whether M1 <= M2 or not
    where Mi is the matroid defined by Hi
    This assumes that the Hi are in reduced form
    """
    S1, S2 = supp(H1[1]), supp(H2[1])
    # Start with loops test
    if not S2 <= S1:
        # the loops of H1 are not all loops of H2
        return False
    # Computes additional loops in H2
    S12 = S1 - S2
    # And remove them from H1
    H1S2 = H1.copy()
    for i in S12:
        H1S2 = remove(H1S2, i)
    # Then we are reduced to compare the PLC with no loops
    X1, P1 = H1S2
    X2, P2 = H2
    if not inf_subs(P1, P2):
        # the 2-pts of H1 are not all 2-pts of H2
        return False

    # We identify 2-pts of H2 in H1
    X1bis = replace(X1, P2)
    if not inf_subs(X1bis[0], X2[0]):
        # not all T2-sets of H1 are a T2-set of H2
        return False

    for l in X1bis[1]:
        if not test_T3(l, X2):
            return False

    return True

def poset_mins_part(Y, LX, f):
    """
    Input:
    - LX is a list of lists X1,...,XN such that only Xi < Y is possible
    - Y is a list of elements
    Output:
    - A list of list Ymin such that contains the minimal elements of Y such that
    the elements of LX \\cup Ymin are not pairwise comparable
    """
    m = len(Y)
    is_minimal = [True] * m

    # First test which elts of Y are killed by elt of X{ind+1},...,XN
    for i in range(m):
        for X in LX:
            if is_minimal[i]:
                for x in X:
                    if f(x, Y[i]):
                        is_minimal[i] = False
                        break

    # Now test inside the elements of X{ind} and Y
    for i in range(m):
        if is_minimal[i]:
            for j in range(i+1, m):
                if is_minimal[j]:
                    if f(Y[j], Y[i]):  # Y[j] <= Y[i]
                        is_minimal[i] = False
                        break
                    elif f(Y[i], Y[j]):  # Y[i] <= Y[j]
                        is_minimal[j] = False

    return [Y[j] for j in range(m) if is_minimal[j]]

def serial_min_poset(LX, f):
    """
    Input:
    - LX is a list of lists X1,...,XN such that each Xi are
    minimal elts (i.e. elts are not pairwise comparable)
    Output:
    - a list LY of the lists Y1,...,YM of the minimal elts of the Xi
    such that the elts of Y1 \\cup...\\cup YM are not pairwise comparable
    """
    N = len(LX)
    Ln = [ len(X) for X in LX ]
    is_minimal = [ [True]*Ln[i] for i in range(N) ]
    for i in range(N):
        for j in range(Ln[i]):
            if is_minimal[i][j]:
                for i1 in range(i+1, N):
                    if is_minimal[i][j]:
                        for j1 in range(Ln[i1]):
                            if is_minimal[i1][j1]:
                                if f(LX[i1][j1], LX[i][j]):
                                    is_minimal[i][j] = False
                                    break
                                elif f(LX[i][j], LX[i1][j1]):
                                    is_minimal[i1][j1] = False
                    else:
                        break

    tmp = [ [ LX[i][j] for j in range(Ln[i]) if is_minimal[i][j] ] for i in range(N) ]
    return [ t for t in tmp if t ]

def split(L):
    N = len(L)//2
    return [L[:N], L[N:]]

def serial_min_merge(LB, LC, f):
    """
    Input:
    - LX is a list of lists X1,...,XN such that:
     * each Xi are minimal elts (i.e. elts are not pairwise comparable)
    Output:
    - a list Y of the minimal elts of the union of the Xi
    """
    N, M = len(LB), len(LC)
    Ln, Lm = [ len(B) for B in LB ], [ len(C) for C in LC ]
    is_minimal = [ [ [True]*Ln[i] for i in range(N) ],
                   [ [True]*Lm[i] for i in range(M) ]]

    for i in range(N):
        for j in range(Ln[i]):
            if is_minimal[0][i][j]:
                for i1 in range(M):
                    if is_minimal[0][i][j]:
                        for j1 in range(Lm[i1]):
                            if is_minimal[1][i1][j1]:
                                if f(LC[i1][j1], LB[i][j]):  # Yind[j] less than Yind[i+n]
                                    is_minimal[0][i][j] = False
                                    break
                                elif f(LB[i][j], LC[i1][j1]):  # Yind[i+n] <= Yind[j]
                                    is_minimal[1][i1][j1] = False
                    else:
                        break
    LB = [ [ LB[i][j] for j in range(Ln[i]) if is_minimal[0][i][j] ] for i in range(N) ]
    LC = [ [ LC[i][j] for j in range(Lm[i]) if is_minimal[1][i][j] ] for i in range(M) ]
    return [ lb for lb in LB if lb ], [lc for lc in LC if lc]

def parallel_min_merge(LB, LC, f, depth, n_procs=1):
    T = 6 # Parameter threshold at which we switch back to non-recursive computations
    nB, nC = sum(map(len,LB)), sum(map(len,LC))
    if nB <= T and nC <= T:
        return serial_min_merge(LB, LC, f)
    elif nB > T and nC > T:
        SLB, SLC = split(LB), split(LC)
        # First parallel split
        if depth < int(log2(n_procs))-1:
            with MyPool(2) as p:
                [[SLB[0], SLC[0]], [SLB[1], SLC[1]]] = p.starmap(
                    partial(parallel_min_merge, f=f, depth=depth+1, n_procs=n_procs),
                    zip(SLB, SLC)
                    )
        else:
            SLB[0], SLC[0] = parallel_min_merge(SLB[0], SLC[0], f, depth, n_procs)
            SLB[1], SLC[1] = parallel_min_merge(SLB[1], SLC[1], f, depth, n_procs)
        # Second parallel split
        if depth < int(log2(n_procs))-1:
            with MyPool(2) as p:
                [[SLB[0], SLC[1]], [SLB[1], SLC[0]]] = p.starmap(
                    partial(parallel_min_merge, f=f, depth=depth+1, n_procs=n_procs),
                    zip(SLB, reversed(SLC))
                    )
        else:
            SLB[0], SLC[1] = parallel_min_merge(SLB[0], SLC[1], f, depth, n_procs)
            SLB[1], SLC[0] = parallel_min_merge(SLB[1], SLC[0], f, depth, n_procs)

        return SLB[0]+SLB[1], SLC[0]+SLC[1]
    elif nB > T and nC <= T:
        SLB = split(LB)
        # No parallel split as C is shared
        SLB[0], LC = parallel_min_merge(SLB[0], LC, f, depth, n_procs)
        SLB[1], LC = parallel_min_merge(SLB[1], LC, f, depth, n_procs)
        return SLB[0]+SLB[1], LC
    else:#nB <= T and nC > T
        SLC = split(LC)
        # No parallel split as C is shared
        LB, SLC[0] = parallel_min_merge(LB, SLC[0], f, depth, n_procs)
        LB, SLC[1] = parallel_min_merge(LB, SLC[1], f, depth, n_procs)
        return LB, SLC[0]+SLC[1]

def parallel_min_poset(LX, f, depth = 0, n_procs = 1):
    T = 8 # Parameter threshold at which we switch back to non-recursive computations
    if sum(map(len,LX)) <= T:
        return serial_min_poset(LX, f)
    LX = split(LX)
    # Parallel split
    if depth < int(log2(n_procs))-1:
        with MyPool(2) as p:
            LX = p.map(partial(parallel_min_poset, f=f, depth=depth+1, n_procs=n_procs), LX)
    else:
        LX = list(map(partial(parallel_min_poset, f=f, depth=depth+1, n_procs=n_procs), LX))
    # sync
    LX = parallel_min_merge(LX[0], LX[1], f, depth, n_procs)
    return LX[0]+LX[1]


# Below we create a variant MyPool of Pool class to allow subprocess
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)
