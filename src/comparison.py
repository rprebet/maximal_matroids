from src.hypergraph import supp, inf_subs, replace, remove, is_inside, diff_size

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