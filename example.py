import src.minimat as minimat

### Data ###
# {1,...,d} is the ground set
# DM[i] are the cyclic flats of rk i of M##
############
# Example: the Vàmos matroid
d = 10
DM = [[],[], [], [{1, 2, 3, 4},{3, 4, 5, 6},{5, 6, 7, 8},{1, 2, 7, 8},{3, 4, 7, 8}]]
############

# Compute upper cover of HT
mL = minimat.minimal_extensions(DM, d, S = [1,2,3,4], v=1, preprocess = False, n_procs=1)

# Printing results
print("Found {} minimal matroids above M:".format(len(mL)))
for i,l in enumerate(mL):
    minimat.printmat(l, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))