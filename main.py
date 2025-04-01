from time import time

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

def rem_sub(XT):
    """
    Input: labeled hypergraph XT (2-list of sets)
    Remove redundant subsets from XT
    Criterion 1: sets that are subsets of another set in XT[k]
    Criterion 2: sets in XT[0] that are one element larger than a set in XT[1]
    This is an in-place function
    """

    # Criterion 1
    for k in range(len(XT)):
        i = 0
        while i < len(XT[k]):
            j = i + 1
            while j < len(XT[k]):
                if XT[k][i] <= XT[k][j]:  # If i-th set is a subset of j-th, remove it
                    XT[k].pop(i)
                    i -= 1  # Adjust index since current i was removed
                    break
                elif XT[k][j] <= XT[k][i]:  # If j-th set is a subset of i-th, remove j-th
                    XT[k].pop(j)
                else:
                    j += 1
            i += 1

    # Criterion 2
    i = 0
    while i < len(XT[0]):
        j, n = 0, len(XT[0][i])
        while j < len(XT[1]):
            if len(XT[1][j]) + 1 == n and XT[1][j] <= XT[0][i]:
                XT[0].pop(i)
                i -= 1  # Adjust index since current i was removed
                break
            j += 1
        i += 1

def solve_int(XT, r):
    """
    Input: labeled hypergraph XT (2-list of sets); r (integer)
    Output: * flag (bool) indicating if modifications have been done
            * HTbis (2-list of sets) fixed hypegraph

    Fix all submodularity failures that require no choice
    depending on the situation given by r 
    (no |circuit|< r included)
    """
    flag = False
    # Fix Case 3
    T1, T2 = XT[0].copy(), XT[1].copy()
    for i in range(len(T2)):
        for j in range(i+1,len(T2)):
            if len(T2[i] & T2[j]) == 1 and not is_inside(T2[i] | T2[j], XT[0]):
                flag = True
                T1.append(T2[i] | T2[j])
                
    HTbis = [T1, T2]
    rem_sub(HTbis)

    if r >= 3:
        # Fix Case 2
        T1, T2 = HTbis
        for i in range(len(T1)):
            for j in range(len(T2)):
                if len(T1[i] & T2[j]) > 1 and not T2[j] <= T1[i]:
                    flag = True
                    T1 = [ T1[k] for k in range(len(T1)) if k != i ] + [T1[i] | T2[j]]
        HTbis = [T1, T2]
        rem_sub(HTbis)

    if r == 3:
        # Fix Case 4
        T1, T2 = HTbis
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) > 1:
                    flag = True
                    T2 = [ T2[k] for k in range(len(T2)) if k not in [i,j] ] + [T2[i] | T2[j]]
        HTbis = [T1, T2]
        rem_sub(HTbis)

    if r >= 4:
        # Fix Case 1
        T1, T2 = HTbis
        for i in range(len(T1)):
            for j in range(i+1,len(T1)):
                S = T1[i] & T1[j]
                if len(S) > 2 and not is_inside(S, T2):
                    flag = True
                    T1 = [ T1[k] for k in range(len(T1)) if k not in [i,j] ] + [T1[i] | T1[j]]
        HTbis = [T1, T2]
        rem_sub(HTbis)

    return flag, HTbis

def detect_mat_case(XT, c):
    """
    Input: labeled hypergraph XT (2-list of sets), c (int)
    Output: c (int), (i,j) (int, int)

    Perform the test of the case 'c' of submodularity fail 
    Return indices (i,j) of corresponding 'c'-failure, if found
    If no failure, return '0, ()' (the hypergraph defines a matroid)
    """

    T1, T2 = XT
    if c == 1:
        for i in range(len(T1)):
            for j in range(i+1,len(T1)):
                S = T1[i] & T1[j]
                if len(S) > 2 and not is_inside(S, T2):
                    return 1, (i,j)
    if c == 2:
        for i in range(len(T1)):
            for j in range(len(T2)):
                if len(T1[i] & T2[j]) > 1 and not T2[j] <= T1[i]:
                    return 2, (i,j)
    if c == 3:
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) == 1 and not is_inside(T2[i] | T2[j], T1):
                    return 3, (i,j)
    if c == 4:
        for i in range(len(T2)):
            for j in range(i+1,len(T2)):
                if len(T2[i] & T2[j]) > 1:
                    return 4, (i,j)
    return 0, () # No failure found

def detect_mat(XT):
    """
    Input: labeled hypergraph XT (2-list of sets)
    Perform submodularity fail tests 
    in a chosen case order
    """
    seq = [4, 2, 1, 3] # Custom case detection order (heuristic)
    for c in seq:
        case, ind = detect_mat_case(XT, c)
        if case != 0:
            return case, ind
    return 0, ()

def replace(L, P, smin):
    """
    Input: hypergraph L (list of sets), partition P (list of sets), smin (int)
    Output: hypergraph L1 (list of sets)

    Replace elements in sets of L base on the equivalence relation
    defined by P, taking minimal element as representative.
    Return in L1, only resulting sets of size >= smin (=[label of L]+1)
    """
    # Construct a dict Lrep to associate each element to its minimal representative
    Prep = [ min(p) for p in P ]
    Lrep = { p:Prep[i] for i in range(len(P)) for p in P[i]}
    # Contruct new hypergraph L1 based on the dict Lrep (repetition is avoided by set structure)
    L1 = [ { Lrep[j] for j in l } for l in L ]
    return [ l for l in L1 if len(l)>=smin ] 

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
    # Perform replacement based on P1 using 'replace', with smin=(label+1)
    XT1 = [ replace(XT[i], P1, 4-i) for i in range(len(XT)) ]
    return [XT1, P1]


def comp_leaves(HT, r, pproc=False):
    # no  cyclic flat of rank < r included
    XT, P = HT
    XT = [XT[0].copy(), XT[1].copy()]
    Lcand = []

    rem_sub(XT)
    if pproc:
        flag = True
        while flag:
            flag, XT = solve_int(XT, r)

    c, ind = detect_mat(XT)
    if c == 0:
      # This rpz a matroid
      Lcand.append([XT, P])

    else:
        T1,  T2 = XT
        if c == 1:
            e1, e2 = T1[ind[0]], T1[ind[1]]

            if r <= 4:
                T1bis = [ T1[i] for i in range(len(T1)) if i not in ind ]
                T1bis.append(e1 | e2)
                Lcand += comp_leaves([[T1bis, T2], P], r, pproc=pproc)

            if r <= 3:
                T2bis = T2.copy()
                T2bis.append(e1 & e2)
                Lcand += comp_leaves([[T1, T2bis], P], r, pproc=pproc)

        elif c == 2:
            e1, e2 = T1[ind[0]], T2[ind[1]]

            if r <= 4:
                T1bis = [ T1[i] for i in range(len(T1)) if i != ind[0] ]
                T1bis.append(e1 | e2)
                Lcand += comp_leaves([[T1bis, T2], P], r, pproc=pproc)

            if r <= 2:
                HTid = identify(HT, e1 & e2)
                Lcand += comp_leaves(HTid, r, pproc=pproc)

        elif c == 3:
            e1, e2 = T2[ind[0]], T2[ind[1]]
            T1bis = T1.copy() + [e1.union(e2)]
            Lcand += comp_leaves([[T1bis, T2], P], r, pproc=pproc)

        elif c == 4:
            e1, e2 = T2[ind[0]], T2[ind[1]]

            if r <= 3:
                T2bis = [ T2[i] for i in range(len(T2)) if i not in ind ]
                T2bis.append(e1 | e2)
                Lcand += comp_leaves([[T1, T2bis], P], r, pproc=pproc)

            if r <= 2:
                HTid = identify(HT, e1 & e2)
                Lcand += comp_leaves(HTid, r, pproc=pproc)

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
                    if len(l) > 4-j:
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
    S = set()
    for l in L:
        S = S.union(l)
    return S

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

def test_T1(l, X2):
    if is_inside(l, X2[0]):
        return True

    l1 = [ l - {e} for e in l ]
    for ll in l1:
        if is_inside(ll, X2[1]):
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
    X1bis = [ replace(X1[i], P2, 4-i) for i in range(2) ]
    if not inf_subs(X1bis[1], X2[1]):
        # not all T2-sets of H1 are a T2-set of H2
        return False

    for l in X1bis[0]:
        if not test_T1(l, X2):
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


def upper_covers(HT, S, v=1, preprocess = False):
    # H = [M, P]
    # M=[M3,M2] is a rank 4 matroid given by its type 3 and 2 sets
    # P <= {1,...,d} is the partition support of M

    XT, P = HT
    d = len(P)
    tleaf, tmin = 0, 0

    v>0 and print("Compute minimals in in " + ", ".join([ ["A","B","C","D"][i-1] for i in S]))

    Lcumins = [ [] for _ in range(4) ]

    if 4 in S:
        for i in range(d):
            for j in range(i+1,d):
                for k in range(j+1, d):
                    for l in range(k+1, d):
                        e1, e2, e3, e4 = [ min(P[ind]) for ind in [i,j,k,l] ]
                        tmp = {e1,e2,e3,e4}
                        if not inter_size(tmp, XT[1],3) and not is_inside(tmp, XT[0]):
                            t = time()
                            tmp = comp_leaves([ [XT[0]+[tmp], XT[1]] , P], 4, pproc=preprocess)
                            t1 = time()
                            Lcumins = poset_mins_part_update(Lcumins, tmp, 4, inf_hyper)
                            tleaf += t1 - t;  tmin += time() - t1
                            v==2 and print("Add {{{},{},{},{}}}: {:.2g}s".format(e1,e2,e3,e4,time()-t))
                            v>2 and print("Add {{{},{},{},{}}}: {:,} cands ({:.2g}s); {:,} D-current mins ({:.2g}s) ".format(e1,e2,e3,e4,len(tmp), t1-t, len(Lcumins[3]), time()-t1))
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))

    if 3 in S:
        for i in range(d):
            for j in range(i+1,d):
                for k in range(j+1, d):
                    e1, e2, e3 = [ min(P[ind]) for ind in [i,j,k] ]
                    lijk = {e1, e2, e3}
                    if not is_inside(lijk, XT[1]):
                        t = time()
                        tmp = comp_leaves([ [ XT[0], XT[1]+[lijk]] , P], 3, pproc=preprocess)
                        t1 = time()
                        Lcumins = poset_mins_part_update(Lcumins, tmp, 3, inf_hyper)
                        tleaf += t1 - t;  tmin += time() - t1
                        v==2 and print("Add {{{},{},{}}}: {:.2g}s".format(e1,e2,e3,time()-t))
                        v>2 and print("Add {{{},{},{}}}: {:,} cands ({:.2g}s); {:,} C-current mins ({:.2g}s) ".format(e1,e2,e3,len(tmp), t1-t, len(Lcumins[2]), time()-t1))
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))

    if 2 in S:
        for i in range(d):
            for j in range(i+1,d):
                e1, e2 = [ min(P[ind]) for ind in [i,j] ]
                t = time()
                tmp = comp_leaves(identify(HT, {e1, e2}), 2, pproc=preprocess)
                t1 = time()
                Lcumins = poset_mins_part_update(Lcumins, tmp, 2, inf_hyper)
                tleaf += t1 - t;  tmin += time() - t1
                v==2 and print("Add {{{},{}}}: {:.2g}s".format(e1,e2,time()-t))
                v>2 and print("Add {{{},{}}}: {:,} cands ({:.2g}s); {:,} B-current mins ({:.2g}s) ".format(e1,e2,len(tmp), t1-t, len(Lcumins[1]), time()-t1))
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))

    if 1 in S:
        t = time()
        tmp = [ remove(HT, min(P[i])) for i in range(d) ]
        t1 = time()
        Lcumins = poset_mins_part_update(Lcumins, tmp, 1, inf_hyper)
        v==2 and print("Add loops: {:.2g}s".format(time()-t))
        v>2 and print("Add loops: {:,} cands ({:.2g}s); {:,} A-current mins ({:.2g}s) ".format(len(tmp), t1-t, len(Lcumins[0]), time()-t1))
        tleaf += t1 - t;  tmin += time() - t1
        v>1 and print("{:,} current mins".format(sum(map(len,Lcumins))))

    v>0 and print("Time elapsed {:.2f}s (total) ; {:.2f}s (cand) ; {:.2f}s (mins)\n".format(tleaf+tmin, tleaf, tmin))
    cumins = []
    for l in Lcumins:
        cumins += l
    return cumins

# Data structure translation
def cyclic_to_partition(CF, d):
    P = [{i} for i in range(1, d+1)]
    HT = [[CF[3].copy(), CF[2].copy()], P]
    for e in CF[0][0]:
        HT = remove(HT, e)
    for L in CF[1]:
        HT = identify(HT, L)
    return HT

def partition_to_cyclic(HT, d):
    CF = []
    XT, P = HT
    loops = set(i for i in range(1, d+1)).difference(supp(P))
    CF.append([{i for i in loops}]  if len(loops)>0 else [])
    CF.append([ {i for i in p} for p in P if len(p)>1 ])
    CF.append(XT[1])
    CF.append(XT[0])
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
    Ci.append("T2: " + " ".join(str(l).replace(" ", "") for l in X[1]))
    Ci.append("T3: " + " ".join(str(l).replace(" ", "") for l in X[0]))
    Ci = [ c for c in Ci if len(c) > 4]
    print(pref+" "+" ; ".join(Ci))


### Tests ####

# Vamos example
### Data ###
XT = [[{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 8}, {1, 2, 7, 8},{3, 4, 7, 8},{1,2,5,6},{1,3,5,7},{8,1,3,6},{8,1,4,5},{1,4,6,7},{8,2,3,5},{2,3,6,7}],[]]
d = 8
P = [{i} for i in range(1, d+1)]
HT = [XT, P]

# Compute upper cover of HT
mL = upper_covers(HT, {1,2,3,4}, v=1, preprocess = False)

# Printing results
print("Found {} minimal matroids above M:".format(len(mL)))
for i,l in enumerate(mL):
    printmat(l, d, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))

