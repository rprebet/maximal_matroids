import src.minimat as minimat

### Data ###
# {1,...,d} is the ground set
# DM[i] are the cyclic flats of rk i of M##
############
# Example: the Vàmos matroid
d = 10
DM = [[],[], [], [{1, 2, 3, 4},{3, 4, 5, 6},{5, 6, 7, 8},{1, 2, 7, 8},{3, 4, 7, 8}]]
#DM = [[],[], [], [{1,2,4,5},{2,3,5,6},{3,4,6,7},{4,5,7,8},{5,6,8,9},{6,7,9,10},{1,7,8,10},{1,2,8,9},{2,3,9,10},{1,3,4,10},{1,2,3,7},{2,3,4,8},{3,4,5,9},{4,5,6,10},{1,5,6,7},{2,6,7,8},{3,7,8,9},{4,8,9,10},{1,5,9,10},{1,2,6,10},{1,3,5,8},{2,4,6,9},{3,5,7,10},{1,4,6,8},{2,5,7,9},{3,6,8,10},{1,4,7,9},{2,5,8,10},{1,3,6,9},{2,4,7,10}]]
############

# Compute upper cover of HT
mL = minimat.minimal_extensions(DM, d, S = [1,2,3,4], v=1, preprocess = False, n_procs=0)

# Printing results
print("Found {} minimal matroids above M:".format(len(mL)))
#for i,l in enumerate(mL):
#    minimat.printmat(l, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))