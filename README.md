# Maximat: a software for efficiently computing maximal matroid degenerations.

This repository contains an open source implementation in Python of algorithms described in the submitted paper:

> *Efficient Algorithms for Maximal Matroid Degenerations and
Irreducible Decompositions of Circuit Varieties*<br>
> E.Liwski and F.Mohammadi and R.Prébet, Apr. 2025<br>
> <https://arxiv.org/abs/2504.16632>

Consider matroids $M$ and $N$ with ground set $\{1,\dotsc,d\}$, given by their list of cyclic flats: `CF(M)` and `CF(N)`, resp..<br>
If $M$ and $N$ have **rank at most 4**, then one can:
* compare $M$ and $N$ with the function call: `inf_cyclic( CF(M), CF(N), d ) `
* compute all maximal degenerations of $M$ with the function call: `maximal_degenerations(CF(M), d)`

All useful functions can be loaded by importing the file **maximat.py** contained in the directory **src/**. The documentation of each function can be obtained either from source code or in th terminal by e.g.:
```python
help(inf_cyclic)
```

The file **example.py** contains an example of usage of this library on the Vàmos matroid. We detail each instruction in the following.
Note that these instructions can be indifferently executed in a script .py or in a terminal.<br>

## A simple example

The file **example.py** starts by loading the main source file as a library:
```python
> import src.maximat as maximat
```
> Note that if you load this file from another location, then you must add before the relative path to the "src" directory.

Then, it defined the input: ground set and cyclic flats listed by increasing rank:
```python
> d = 9
> DM = [[],[], [], [{1,2,3,4},{3,4,5,6},{5,6,7,8},{1,2,7,8},{3,4,7,8}]]
```

Then, it computes all maximal degenerations below the matroid encoded by `DM`, and stores them in `mL`.
```python
> mL = maximat.maximal_degenerations(DM, d, S = [1,2,3,4], v=1, preprocess = False, n_procs=1)

Compute maximals in S1(M), S2(M), S3(M), S4(M) (1 process)
Time elapsed 0.07s (total) ; 0.04s (max cands) ; 0.03s (inter maxs)
```
Here, we set the optional parameters:
* `S=[1,2,3,4]` to compute degenerations in $S_1(M)$, $S_2(M)$, $S_3(M)$ and $S_4(M)$;
* `v=1` we print only global information about the computations : the $S_i(M)$ in which we compute and the timings:
    * `(total)`: total elapsed time;
    * `(max cands)`: time spent computing maximal degenerations when adding 1 new dependancy;
    * `(inter maxs)`: time spent interreduce to the maximal degenerations among the above ones.
* `preprocess=False` an additional feature that might remove some useless recursive calls during computation. Since this does not change the timings that much, we generally do not use it.
* `n_procs=1` (default 1) is the maximum number of processes to use for parallelization. If n_procs <= 0 it uses automatically all available cpus.

Then, we display the result using a tailored function for nice printing.
```python
> print("Found {} maximal matroids below M:".format(len(mL)))
> for i,l in enumerate(mL):
>    maximat.printmat(l, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))

Found 62 maximal matroids below M:
M1 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,5,6}
M2 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,5,9}
M3 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,6,9}
M4 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,5,7}
M5 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,3,5}
M6 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,5,9}
M7 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,6,7}
M8 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,3,6}
M9 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,6,9}
M10| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,9,7}
...
M59| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {3,4,7,8,9}
M60| T3: {1,2,3,4} {3,4,5,6} {8,1,2,7} {8,3,4,7} {5,6,7,8,9}
M61| T1: {3,4} ; T3: {8,5,6,7} {8,1,2,7}
M62| T1: {8,7} ; T3: {1,2,3,4} {3,4,5,6}
```

For each matroid in mL, we display its cyclic flats, grouped by rank. More precisely, after each `Ti:` all cyclic flats of rank i are displayed.