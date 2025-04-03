import minimat
import datamat

# Vamos example
### Data ###
XT = [[], [{1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 8}, {1, 2, 7, 8},{3, 4, 7, 8}]]
d = 12
P = [{i} for i in range(1, d+1)]
HT = [XT, P]

# Compute upper cover of HT
mL = minimat.upper_covers(HT, {1,2,3,4}, v=1, preprocess = False)

# Printing results
print("Found {} minimal matroids above M:".format(len(mL)))
for i,l in enumerate(mL):
    datamat.printmat(l, d, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))

