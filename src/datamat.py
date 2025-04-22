from src.hypergraph import remove, identify, supp

# Data structure translation
def cyclic_to_partition(CF, d):
    # Compute a reduced labeled hypergraph associated to cyclic flats given in CF with ground set [d]
    # where CF[i] are the cyclic flats of rank i
    assert len(CF) == 4 and all(type(t)==list and len(t)==0 or type(t[0])==set for t in CF), "Wrong input for CF"
    assert d > 0 and all( s <= d for cf in CF for c in cf for s in c ), "Wrong input for d"

    P = [{i} for i in range(1, d+1)]
    HT = [[CF[2].copy(), CF[3].copy()], P]
    if CF[0]:
        for e in CF[0][0]:
            HT = remove(HT, e) # Remove loops
    for L in CF[1]:
        HT = identify(HT, L) # Identify double points
    return HT

def partition_to_cyclic(HT, d):
    # Compute the cyclic flats encoded in the reduced hypergraph HT with ground set [d]
    assert len(HT) == 2 and len(HT[0]) == 2 and all(type(t)==list and len(t)==0 or type(t[0])==set for t in HT[0]) and all(type(t)==set for t in HT[1]), "Wrong input for HT"
    assert d > 0 and all( s <= d for cf in HT[0] for c in cf for s in c ) and all( c <= d for cf in HT[1] for c in cf), "Wrong input for d"

    CF = []
    XT, P = HT
    loops = set(i for i in range(1, d+1)).difference(supp(P))
    CF.append([{i for i in loops}]  if len(loops)>0 else []) # Type 0
    CF.append([ {i for i in p} for p in P if len(p)>1 ]) # Type 1
    CF.append(XT[0]) # Type 2
    CF.append(XT[1]) # Type 3
    return CF

# Display function
def printmat(CF, pref=""):
    # Nice print of the matroids defined the cyclic flats
    # after "Ti:" are listed all rank i cyclic flats
    assert len(CF) == 4 and all(type(t)==list and len(t)==0 or type(t[0])==set for t in CF), "Wrong input for CF"

    Ci = []
    len(CF[0])>0 and Ci.append("T0: {" + ",".join( str(i) for i in CF[0]) + "}")
    Ci.append("T1: " + " ".join(( "{"+",".join(str(i) for i in p)+"}" for p in CF[1] if len(p)>1 )))
    Ci.append("T2: " + " ".join(str(l).replace(" ", "") for l in CF[2]))
    Ci.append("T3: " + " ".join(str(l).replace(" ", "") for l in CF[3]))
    Ci = [ c for c in Ci if len(c) > 4]
    print(pref+" "+" ; ".join(Ci))