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

def supp(L):
    return set.union(*L) if L else set()
