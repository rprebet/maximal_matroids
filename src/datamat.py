from src.hypergraph import remove, identify, supp

# Data structure translation
def cyclic_to_partition(CF, d):
    # Compute a reduced labeled hypergraph associated to CF
    P = [{i} for i in range(1, d+1)]
    HT = [[CF[2].copy(), CF[3].copy()], P]
    if CF[0]:
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
def printmat(CF, d, pref=""):
    # Nice print of the matroids defined the cyclic flats
    Ci = []
    len(CF[0])>0 and Ci.append("T0: {" + ",".join( str(i) for i in CF[0]) + "}")
    Ci.append("T1: " + " ".join(( "{"+",".join(str(i) for i in p)+"}" for p in CF[1] if len(p)>1 )))
    Ci.append("T2: " + " ".join(str(l).replace(" ", "") for l in CF[2]))
    Ci.append("T3: " + " ".join(str(l).replace(" ", "") for l in CF[3]))
    Ci = [ c for c in Ci if len(c) > 4]
    print(pref+" "+" ; ".join(Ci))