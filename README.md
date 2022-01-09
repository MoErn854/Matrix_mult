# Matrix_mult
Small Application for matrix multiplication using different algorithms.

### Explanation 
#### I. Naive Approach without using divide and conquer:

Explanation:  <br>

![image](https://user-images.githubusercontent.com/84231705/148701108-ced52c10-ff77-4005-b92c-f2b4801c412c.png)

Pseudo-code:
```
matrix_mult_naive(A, B, n):
1 SET product → matrix of zeros
2 for i to n:
3   for j to n:
4     for k to n:
5       product[i, j] += A[i, k] * B[k, j]
6 return product
```

<hr>

#### II. Naive Approach using divide and conquer:
<br>
Explanation:
<br>
![image](https://user-images.githubusercontent.com/84231705/148701172-30ff5dea-d54e-4d5d-a977-f9d4e49764b8.png)
<br>
Pseudo-code:
<br>
![image](https://user-images.githubusercontent.com/84231705/148701192-4a76ba3b-7a82-485d-9a56-275236837d3d.png)
<br>
time complixity :
<br>
![image](https://user-images.githubusercontent.com/84231705/148701211-5921902f-e0cf-4f52-acd2-7083f47f180d.png)


⸪ 𝑛/2𝑖 = 1 
⸫ 2^𝑖 = 𝑛
⸫ 𝑖 = log2 𝑛
<br>
Total Cost = Σ 𝑘 ∗ 8𝑖 ∗ (𝑛/2𝑖) = 𝑘 𝑛 Σ 4^𝑖 = 𝑘𝑛 + 4 𝑘𝑛 + ⋯+ 𝒌𝒏^𝟑 <br>
**O(n3)**

<hr>

#### III. Strassen’s Approach not using divide and conquer:

Explanation:
<br>
![image](https://user-images.githubusercontent.com/84231705/148701318-38d41673-9dd7-434a-b106-254b48c9dc36.png)
<br>

Pseudo-code:
```
matrix_mult_fast (A,B, sizeN) :
1 A, B = add_zeros_matrix_power_two(A, B)
2 let product be a new n×n matrix
3
4 if len(A) == 1
5     return A ⋅ B
6 else
7     N = len(A) / 2
8     partition A, and B
9
10    M1 = matrix_mult_fast(A11+A22, B11+B22, N)
11    M2 = matrix_mult_fast(A21+A22, B11, N)
12    M3 = matrix_mult_fast(A11, B12-B22, N)
13    M4 = matrix_mult_fast(A22, B21- B11, N)
14    M5 = matrix_mult_fast(A11 + A12, B22, N)
15    M6 = matrix_mult_fast(A21-A11, B11+B12, N)
16    M7 = matrix_mult_fast(A12-A22, B21+B22, N) 
17 
18    product11 = M1 + M4 - M5 + M7 
19    product12 = M3 + M5 
20    product21 = M2 + M4 
21    product22 = M1 – M2 + M3 + M6 
22 
23 return product[ : sizeN, : sizeN]
```

time complixity : 
![image](https://user-images.githubusercontent.com/84231705/148701404-2fdd7668-6d24-4c6d-91dc-a62dfc700e1d.png)


⸪ 𝑛/2𝑖 = 1 
⸫ 2^𝑖 = 𝑛
⸫ 𝑖 = log2 𝑛

Total Cost = Σ 𝑘 ∗ 7𝑖 ∗ (𝑛/2𝑖) = 𝑘 * 𝑛 Σ (7^𝑖/2^i) = 𝑘𝑛 + ... + 𝒌𝒏^log2(7) <br>
**O(n^2.81)**

<hr>

#### References: <br>
I. Eli Bendersky's website. 2015. Visualizing matrix multiplication as a linear combination. [ONLINE] Available at: https://eli.thegreenplace.net/2015/visualizing-matrix-multiplication-as-a-linear-combination/ <br>
II. Geeks for Geeks. 2021. Python Program to find whether a no is power of two. [ONLINE] Available at: https://www.geeksforgeeks.org/python-program-to-find-whether-a-no-is-power-of-two/<br>
III. Geeks for Geeks. 2021. Strassens Matrix Multiplication. [ONLINE] Available at: https://www.geeksforgeeks.org/strassens-matrix-multiplication/ <br>
IV. Shiva Thudi. 2017. Matrix multiplication using the Divide and Conquer paradigm. [ONLINE] Available at: https://shivathudi.github.io/jekyll/update/2017/06/15/matr-mult.html. <br>
V. NumPy. 2021. numpy.array_equal. [ONLINE] Available at: https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html <br>
VI. Wikipedia. Strassen algorithm. [ONLINE] Available at: https://en.wikipedia.org/wiki/Strassen_algorithm <br>
