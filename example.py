import src.maximat as maximat

### Data ###
# {1,...,d} is the ground set
# DM[i] are the cyclic flats of rk i of M##
############
# Example: the Vàmos matroid
d = 9
DM = [[],[], [], [{1, 2, 3, 4},{3, 4, 5, 6},{5, 6, 7, 8},{1, 2, 7, 8},{3, 4, 7, 8}]]
############

# Compute lower cover of HT
mL = maximat.maximal_degenerations(DM, d, S = [1,2,3,4], v=1, preprocess = False, n_procs=1)

# Printing results
print("Found {} maximal matroids below M:".format(len(mL)))
for i,l in enumerate(mL):
    maximat.printmat(l, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))
