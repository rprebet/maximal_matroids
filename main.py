from time import time
from itertools import combinations

### Code ###

# Operation on hypergraphs
def is_inside(s, L):
    """
    Input: vertex s (set); hypergraph L (list of sets)
    Check if s is contained in an element of L
    """
    for l in L:
        if s <= l: 
            return True
    return False

def inter_size(s, L, n):
    """
    Input: vertex s (set); hypergraph L (list of sets); n (int)
    Check if s has an intersection with an element of L of size >= n
    (this also means that a n-subset of s is contained in some l in L)
    """
    for l in L:
        if len(l & s) >= n:
            return True
    return False


def diff_size(s, L, n):
    """
    Input: vertex s (set); hypergraph L (list of sets); n (int)
    Check if there is l in L s.t. (s - l) has size == n
    (this also means that a (|s|-1)-subset of s is contained in l)
    """
    for l in L:
        if len(s - l) == n:
            return True
    return False

from copy import deepcopy

def rem_sub(XT):
    """
    Input: labeled hypergraph XT (2-list of sets)
    Remove redundant subsets from XT to satisfy the
    axioms of labeled hypergraphs.
    """
    XT_new = [XT[0].copy(), XT[1].copy()]  # Avoid modifying the input

    # Criterion 1: Remove sets that are subsets of another set in XT[k]
    for k in range(len(XT_new)):
        XT_new[k] = [
            s for i, s in enumerate(XT_new[k]) 
            if not any(s <= XT_new[k][j] for j in range(len(XT_new[k])) if i != j)
        ]

    # Criterion 2: Remove sets in XT[1] that are one element larger than a set in XT[0]
    if len(XT_new) > 1:
        XT_new[1] = [
            s for s in XT_new[1] 
            if not any(len(t) + 1 == len(s) and t <= s for t in XT_new[0])
        ]

    return XT_new

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

def replace(XT, P):
    """
    Input: labeled hypergraph XT (2-list of sets), partition P (list of sets)
    Output: labeled hypergraph XT1 (2-list of sets)

    Replace elements in sets of each XT[i] based on the equivalence relation
    defined by P, taking minimal element as representative.
    Return in XT1, only resulting sets of size >= 3+i=(Type+1)
    """
    # Construct a dict Lrep to associate each element to its minimal representative
    Lrep = { elem: min(p) for p in P for elem in p}
    # Contruct new hypergraph L1 based on the dict Lrep (repetition is avoided by set structure)
    XT1 = [ [ { Lrep[j] for j in l } for l in L ] for L in XT ]
    return [[l for l in L if len(l) >= 3 + i] for i, L in enumerate(XT1)]

def identify(HT, l):
    """
    Input:  * HT = (XT, P), labeled hypergraph HT (2-list of sets), partition P (list of sets);
            * points to identify l (set)
    Identify all elements of L in HT (make them double points).
    Call 'replace' on the corresponding new partition
    """
    # Construct the new partition P1
    # S is the eq. class of elements in L based on existing partition P
    XT, P = HT
    P1, S = [], set({})
    for p in P:
        if len(l & p)>0:
            S = S | p
        else:
            P1.append(p)
    P1.append(S)
    # Perform replacement based on P1 using 'replace'
    XT1 = replace(XT, P1)
    return [XT1, P1]



def comp_leaves(HT, r, pproc=False):
    # no  cyclic flat of rank < r included
    XT, P = HT
    XT = [XT[0].copy(), XT[1].copy()]
    Lcand = []

    # We first reduce the hypergraph as we added some edges
    XT = rem_sub(XT)
    if pproc:
        flag = True
        while flag:
            flag, XT = solve_int(XT, r)

    c, ind = detect_mat(XT, r)
    if c == 0:
      # This rpz a matroid
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

def remove(HT, e):
    # Set e as a loop
    XT, P = HT
    i = 0
    while i < len(P) and not e in P[i]:
        i += 1
    if i == len(P):
        return HT

    if len(P[i]) == 1:
        P1 = P[:i]+P[i+1:]
        XT1 = [[], []]
        for j in range(2):
            for l in XT[j]:
                if e in l:
                    if len(l) > 3+j:
                        XT1[j].append(l - {e})
                else:
                    XT1[j].append(l)

        return [XT1, P1]

    newPi = P[i] - {e}
    P1 = P[:i] + [newPi] + P[i+1:]
    if min(P[i]) != e:
        return [XT, P1]

    e1 = min(newPi)
    XT1 = [[], []]
    for j in range(2):
        for l in XT[j]:
            if e in l:
                XT1[j].append(l- {e} | {e1})
            else:
                XT1[j].append(l)

    return [XT1, P1]

def supp(L):
    return set.union(*L) if L else set()

def inf_subs(P1,P2):
    # Test whether every elt of P1 is inside an elt of P2
    for p1 in P1:
        flag = True
        for p2 in P2:
            if p1 <= p2:
                flag = False
                break
        if flag:
            return False
    return True

def test_T3(l, X2):
    if is_inside(l, X2[1]):
        return True
    if diff_size(l, X2[0], 1):
        return True
    return False

def inf_hyper(H1, H2):
    # Test whether M1 <= M2 or not
    # where Mi is the matroid defined by Hi
    # This assumes that the Hi are in reduced form
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

def poset_mins_part_update(LX, Y, ind, f):
    """
    Input:
    - LX is a list of lists X1,...,XN such that:
     * only X1>....>XN is possible
     * X1 \\cup ... \\cup XN are minimal elts (i.e. elts are not pairwise comparable)
    - Y is a list of elements
    - 1 <= ind <= N an integer such that Y>Xj only if j>=ind
    Output:
    - A list of list X1,...,Yi,...,XN such that:
     * only X1>..>Yind>..>XN is possible
     * X1 \\cup..Yind \\cup...\\cup XN are the minimal elts of
       X1 \\cup ... \\cup XN \\cup Y
    """

    N = len(LX)
    Ln = [ len(X) for X in LX ]
    m, n = len(Y), Ln[ind-1]
    Yind = LX[ind-1]+Y
    is_minimal = [ [True]*Ln[i] for i in range(ind-1) ] + [ [True] * (n + m) ]


    # First test which elts of Y are killed by elt of X{ind+1},...,XN
    for i in range(m):
        x = Y[i]
        for k in reversed(range(ind, N)):
            if is_minimal[ind-1][i+n]:
                for j in range(Ln[k]):
                    if f(LX[k][j], Y[i]):
                        is_minimal[ind-1][i+n] = False
                        break

    # Now test inside the elements of X{ind} and Y
    for i in range(m):
        if is_minimal[ind-1][i+n]:
            for j in range(n+i):
                if is_minimal[ind-1][j]:
                    if f(Yind[j], Yind[i+n]):  # Yind[j] less than Yind[i+n]
                        is_minimal[ind-1][i+n] = False
                        break
                    elif f(Yind[i+n], Yind[j]):  # Yind[i+n] <= Yind[j]
                        is_minimal[ind-1][j] = False

    # Finally test if the remaining Y kills elts in X1,...,X{ind-1}
    for i in range(m):
        x = Y[i]
        if is_minimal[ind-1][i]:
            for k in reversed(range(ind-1)):
                for j in range(Ln[k]):
                    if f(x, LX[k][j]):
                        is_minimal[k][j] = False

    Lmins = [ [LX[k][j] for j in range(Ln[k]) if is_minimal[k][j]] for k in range(ind-1) ]
    Lmins.append([Yind[j] for j in range(n+m) if is_minimal[ind-1][j]])
    Lmins += LX[ind:]

    return Lmins

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
        return remove(HT, min(P[i]))
    if i == 2:
        return identify(HT, x)
    if i == 3:
        return [ [XT[0]+[x], XT[1]] , P]
    if i == 4:
        return [ [XT[0], XT[1]+[x]] , P]

def upper_covers(HT, S, v=1, preprocess = False):
    # H = [X, P]
    # X=[T2, T3] is a labeled hypergraph given by its type 2 and 3, representing a rank 4 simple matroid
    # P <= {1,...,d} is the partition support of M

    XT, P = HT
    d = len(P)
    tleaf, tmin = 0, 0

    v>0 and print("Compute minimals in " + ", ".join(["S{}(M)".format(i) for i in S]))

    Lcumins = [ [] for _ in range(4) ]

    for i in sorted(S, reverse=True): # We start by highest Si, being smallest
        for Lind in combinations(range(d), i): # Take a circuit of size i
            x = { min(P[ind]) for ind in Lind } # Its representative
            if not redund_edge(XT, x, i): # Check x is not redundant in XT
                t = time()
                cands = comp_leaves(add_edge(HT, x, i), i, pproc=preprocess) if i>1 else [add_edge(HT, x, 1)]
                t1 = time()
                Lcumins = poset_mins_part_update(Lcumins, cands, i, inf_hyper)
                tleaf += t1 - t;  tmin += time() - t1
                v==2 and print("Add {}: {:.2g}s".format(str(x).replace(" ", ""),time()-t))
                v>2 and print("Add {}: {:,} cands ({:.2g}s); {:,} S{}-current mins ({:.2g}s) ".format(str(x).replace(" ", ""),len(cands), t1-t, len(Lcumins[i-1]), i, time()-t1))
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))


    v>0 and print("Time elapsed {:.2f}s (total) ; {:.2f}s (cand) ; {:.2f}s (mins)\n".format(tleaf+tmin, tleaf, tmin))
    cumins = []
    for l in Lcumins:
        cumins += l
    return cumins

# Data structure translation
def cyclic_to_partition(CF, d):
    # Compute a reduced labeled hypergraph associated to CF
    P = [{i} for i in range(1, d+1)]
    HT = [[CF[2].copy(), CF[3].copy()], P]
    for e in CF[0][0]:
        HT = remove(HT, e) # Remove loops
    for L in CF[1]:
        HT = identify(HT, L) # Identify double points
    return HT

def partition_to_cyclic(HT, d):
    # Inverse the reduction of a reduced labeled hypergraph
    CF = []
    XT, P = HT
    loops = set(i for i in range(1, d+1)).difference(supp(P))
    CF.append([{i for i in loops}]  if len(loops)>0 else []) # Type 0
    CF.append([ {i for i in p} for p in P if len(p)>1 ]) # Type 1
    CF.append(XT[0]) # Type 2
    CF.append(XT[1]) # Type 3
    return CF

# Display function
def printmat(H, d, pref=""):
    # Nice print of the matroids defined by H
    # d defines the ground set
    Ci = []
    X, P = H
    loops = set(i for i in range(1, d+1)).difference(supp(P))
    len(loops)>0 and Ci.append("T0: {" + ",".join( str(i) for i in loops) + "}")
    Ci.append("T1: " + " ".join(( "{"+",".join(str(i) for i in p)+"}" for p in P if len(p)>1 )))
    Ci.append("T2: " + " ".join(str(l).replace(" ", "") for l in X[0]))
    Ci.append("T3: " + " ".join(str(l).replace(" ", "") for l in X[1]))
    Ci = [ c for c in Ci if len(c) > 4]
    print(pref+" "+" ; ".join(Ci))


### Tests ####

# Vamos example
### Data ###
XT = [[], [{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 8}, {1, 2, 7, 8},{3, 4, 7, 8}]]
d = 9
P = [{i} for i in range(1, d+1)]
HT = [XT, P]

#import cProfile
# Compute upper cover of HT
#cProfile.run('mL = upper_covers(HT, {1,2,3,4}, v=1, preprocess = False)')
mL = upper_covers(HT, {1,2,3,4}, v=1, preprocess = False)

# Printing results
print("Found {} minimal matroids above M:".format(len(mL)))
for i,l in enumerate(mL):
    printmat(l, d, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))

