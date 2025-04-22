# Minimat: a software for efficiently computing minimal extension matroids.

This repository contains an open source implementation in Python of algorithms described in the submitted paper:

> *Efficient Algorithms for Minimal Matroid Extensions and Irreducible Decompositions of Circuit Varieties*<br>
> E.Liwski and F.Mohammadi and R.Prébet, Apr. 2025

The functions implemented mainly allows one to:
* compare two matroids of rank at most 4 with the function **inf_cyclic**
* compute all minimal extensions of with the function **minimal_extensions**

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

Then, it computes all minimal extensions above the matroid encoded by DM, and stores them in mL.
```python
> mL = minimat.minimal_extensions(DM, d, S = [1,2,3,4], v=1, preprocess = False)

Compute minimals in S1(M), S2(M), S3(M), S4(M)
Time elapsed 0.06s (total) ; 0.01s (cand) ; 0.05s (mins)
```
Here, we set the optional parameters:
* **S=[1,2,3,4]** to compute extensions in S_1, S_2, S_3 and S_4;
* **v=1** we print only global information about the computations : the S_i in which we compute and the timings:
    * (tot): total elapsed time;
    * (cand): time spent computing extensions;
    * (mins): time spent identifying the minimal ones.
* **preprocess=False** an additional feature that might remove some useless recursive calls during computation. Since this does not change the timings that much, we generally do not use it.

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
M11| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,5,9}
M12| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,6,7}
M13| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,3,6}
M14| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,6,9}
M15| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,3,9,7}
M16| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,3,9}
M17| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,4,5,7}
M18| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,4,5}
M19| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,4,5,9}
M20| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,4,6,7}
M21| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,4,6}
M22| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,4,6,9}
M23| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,4,9,7}
M24| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,4,9}
M25| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,5,6,9}
M26| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,5,9,7}
M27| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,5,9}
M28| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {1,9,6,7}
M29| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,1,6,9}
M30| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {2,3,5,7}
M31| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,2,3,5}
M32| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,3,5}
M33| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {2,3,6,7}
M34| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,2,3,6}
M35| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,3,6}
M36| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,3,7}
M37| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,2,3}
M38| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {2,4,5,7}
M39| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,2,4,5}
M40| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,4,5}
M41| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {2,4,6,7}
M42| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,2,4,6}
M43| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,4,6}
M44| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,4,7}
M45| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,2,4}
M46| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,5,6}
M47| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,5,7}
M48| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,2,5}
M49| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,2,6,7}
M50| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,2,6}
M51| T3: {1,2,3,4} {8,1,2,7} {3,4,5,6,7,8}
M52| T3: {1,2,3,4} {8,5,6,7} {8,1,2,7} {8,3,4,7} {3,4,5,6,9}
M53| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {3,4,7,8,9}
M54| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,3,5,7}
M55| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,3,5}
M56| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,3,6,7}
M57| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,3,6}
M58| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,4,5,7}
M59| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,4,5}
M60| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {9,4,6,7}
M61| T3: {1,2,3,4} {3,4,5,6} {8,5,6,7} {8,1,2,7} {8,3,4,7} {8,9,4,6}
M62| T3: {1,2,3,4} {3,4,5,6} {8,1,2,7} {8,3,4,7} {5,6,7,8,9}
```
