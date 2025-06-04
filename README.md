# Minimat: a software for efficiently computing minimal extension matroids.

This repository contains an open source implementation in Python of algorithms described in the submitted paper:

> *Efficient Algorithms for Minimal Matroid Extensions and Irreducible Decompositions of Circuit Varieties*<br>
> E.Liwski and F.Mohammadi and R.Prébet, Apr. 2025<br>
> <https://arxiv.org/abs/2504.16632>

Consider matroids $M$ and $N$ with ground set $\{1,\dotsc,d\}$, given by their list of cyclic flats: `CF(M)` and `CF(N)`, resp..<br>
If $M$ and $N$ have **rank at most 4**, then one can:
* compare $M$ and $N$ with the function call: `inf_cyclic( CF(M), CF(N), d ) `
* compute all minimal extensions of $M$ with the function call: `minimal_extensions(CF(M), d)`

All useful functions can be loaded by importing the file **minimat.py** contained in the directory **src/**. The documentation of each function can be obtained either from source code or in th terminal by e.g.:
```python
help(inf_cyclic)
```

The file **example.py** contains an example of usage of this library on the Vàmos matroid. We detail each instruction in the following.
Note that these instructions can be indifferently executed in a script .py or in a terminal.<br>

## A simple example

The file **example.py** starts by loading the main source file as a library:
```python
> import src.minimat as minimat
```
> Note that if you load this file from another location, then you must add before the relative path to the "src" directory.

Then, it defined the input: ground set and cyclic flats listed by increasing rank:
```python
> d = 9
> DM = [[],[], [], [{1,2,3,4},{3,4,5,6},{5,6,7,8},{1,2,7,8},{3,4,7,8}]]
```

Then, it computes all minimal extensions above the matroid encoded by `DM`, and stores them in `mL`.
```python
> mL = minimat.minimal_extensions(DM, d, S = [1,2,3,4], v=1, preprocess = False, n_procs=1)

Compute minimals in S1(M), S2(M), S3(M), S4(M) (1 process)
Time elapsed 0.07s (total) ; 0.04s (min cands) ; 0.03s (inter mins)
```
Here, we set the optional parameters:
* `S=[1,2,3,4]` to compute extensions in $S_1(M)$, $S_2(M)$, $S_3(M)$ and $S_4(M)$;
* `v=1` we print only global information about the computations : the $S_i(M)$ in which we compute and the timings:
    * `(total)`: total elapsed time;
    * `(min cands)`: time spent computing minimal extensions when adding 1 new dependancy;
    * `(inter mins)`: time spent interreduce to the minimal extensions among the above ones.
* `preprocess=False` an additional feature that might remove some useless recursive calls during computation. Since this does not change the timings that much, we generally do not use it.
* `n_procs=1` (default 1) is the maximum number of processes to use for parallelization. If n_procs <= 0 it uses automatically all available cpus.

Then, we display the result using a tailored function for nice printing.
```python
> print("Found {} minimal matroids above M:".format(len(mL)))
> for i,l in enumerate(mL):
>    minimat.printmat(l, pref="M{: <{width}}|".format(i+1, width=len(str(len(mL)))))

Found 62 minimal matroids above M:
M1 | T1: {3,4} ; T3: {8,5,6,7} {8,1,2,7}
M2 | T1: {8,7} ; T3: {1,2,3,4} {3,4,5,6}
M3 | T3: {3,4,5,6} {8,5,6,7} {1,2,3,4,7,8}
M4 | T3: {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,3,4,9}
M5 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,5,6}
M6 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,5,9}
M7 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,2,6,9}
M8 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,3,4,7} {1,2,7,8,9}
M9 | T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,5,7}
M10| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,3,5}
...
M59| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,4,5}
M60| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,4,6,7}
M61| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,4,6}
M62| T3: {1,2,3,4} {3,4,5,6} {8,1,2,7} {8,3,4,7} {5,6,7,8,9}
```

For each matroid in mL, we display its cyclic flats, grouped by rank. More precisely, after each `Ti:` all cyclic flats of rank i are displayed.