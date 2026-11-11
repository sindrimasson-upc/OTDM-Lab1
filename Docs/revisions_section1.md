# Suggested revisions to Section 1 (lambda = 0)

Each item gives the text to find in `main.tex` and its replacement, with the reason.
Numbers were checked against `uo_nn_batch_st.csv`; everything else in Section 1
(table values, means, times, ratios, TPR/TNR counts) matched.

---

## 0. Introduce the code and the random line search in the setup (apply first)

The report uses numbers that the batch does not produce. The script that produces them, and
the fact that the line search is random, should be stated where the setup is described, so
that both Section 2 and Section 3 can rely on it. `sec_lambda.tex` already assumes this
paragraph exists (it refers to Section 1 for both points).

Find (end of Table 1 in "Experimental setup"):
```
\label{tab:setup}
\end{table}
```
Replace:
```
\label{tab:setup}
\end{table}

\textbf{Code.} The batch script \texttt{uo\_nn\_\allowbreak batch\_st.m} runs the 90 instances with our implementation (\texttt{uo\_nn\_\allowbreak solve\_st.m} and \texttt{uosol\_st.m}) and produces the iterations, final losses, gradient norms, times and accuracies. The script \texttt{uo\_nn\_\allowbreak lambda\_\allowbreak analysis.m} repeats the runs with the same settings and records the quantities that the batch does not write: the gradient norm, the strong Wolfe outcome, $\kappa(H_k)$ and $\cos\theta_k$ at every iteration, the gradient norm at the SGM solutions, $\|w^*\|$, the Hessian of $\tilde L$ at $w^*$, and the true and false positives on the test set, also split by the true digit of each image. The true digit is known because the test images are the same for every target; only the labels change.

\textbf{Random line search.} The line search \texttt{uoBLSNW32} draws random numbers from the global Matlab generator, so the accepted step lengths, and therefore the number of GM and BFGS iterations, depend on the state of the generator. In the batch this state is fixed because \texttt{uo\_nn\_dataset} resets it just before every run, so the batch is reproducible. To see how much the iteration counts depend on it, \texttt{uo\_nn\_\allowbreak lambda\_\allowbreak analysis.m} repeats every GM and BFGS run with five other states (\texttt{rng(1)} to \texttt{rng(5)}). For example, BFGS on digit 1 with $\lambda=0$ needed between 4 and 22 iterations. Differences of a few iterations between two single runs are therefore not meaningful.
```

When your partner adds their own extra scripts (the 1000-iteration reruns, the
loss/gradient timings, the figure `fig_rk_lambda0.png`), name them in the **Code**
paragraph as well.

## 1. (Optional) Median GM iterations (Section 1.2, GM bullet)

Correct as written: 43.5 is the median over the 8 digits where GM converged, which is how
the sentence reads. Only if you want it side by side with the BFGS median (23.5, over all
10 digits): the GM median over all 10 is (61+66)/2 = 63.5. No change needed otherwise.

## 2. Definition of global convergence (Section 1.2)

Zoutendijk's theorem gives ||grad L(w^k)|| -> 0, not convergence of the iterates
themselves, and it needs a Lipschitz continuous gradient.

Find:
```
An algorithm is globally convergent if the iterates converge to a stationary point of $\tilde L$ ($\nabla\tilde L(w^*)=0$) from any starting point. For a smooth function that is bounded below, GM and BFGS are globally convergent if
```
Replace:
```
An algorithm is globally convergent if $\lim_{k\to\infty}\|\nabla\tilde L(w^k)\|=0$ from any starting point, that is, the iterates approach stationary points of $\tilde L$. For a function that is bounded below and has a Lipschitz continuous gradient, GM and BFGS are globally convergent (Zoutendijk's theorem) if
```

## 3. Typo (end of the bullet list in Section 1.2)

Find:
```
The two GM failures within 100 iterations are caused by the iteration limit and not by a lack of convergence, implying GM requires more iterationto converge than BFGS.
```
Replace:
```
The two GM failures within 100 iterations are caused by the iteration limit and not by a lack of convergence: GM simply needs more iterations to converge than BFGS.
```

## 4. Table 2 caption: the values come from the CSV, and its iteration column is one higher

Find:
```
$t$ is rounded to two significant digits in the log.}
```
Replace:
```
$t$ is rounded to two significant digits in the CSV output. The iteration column of the CSV counts the final iterate as well and is one higher.}
```

## 5. Explanation of why 8 is hardest (Section 1.4, "Across digits")

"8 fills the most pixels" does not hold up: from the clean 7x5 patterns, 5 also has
17 lit pixels (counts for 1..9,0: 10 14 14 15 17 16 11 17 16 16) and 5 is recognised
perfectly. What does explain the false positives on 8 is the overlap: all 14 pixels
of a 3 are lit in an 8, a 9 shares 15 of 16, a 0 shares 14 of 16 and a 6 shares 13 of 16.

Find:
```
However, considering that the digit 8 fills the most pixels out of all numbers, it seems plausible that it may be the most difficult one to learn.
```
Replace:
```
The clean $7\times5$ patterns suggest an explanation. Every lit pixel of a 3 is also lit in an 8, a 9 shares 15 of its 16 lit pixels with an 8, and a 0 and a 6 share 14 and 13 of their 16. After blurring, these digits are therefore hard to tell apart from an 8. The number of lit pixels alone does not explain it, since 5 has as many lit pixels as 8 (17) and is recognised perfectly. The same overlap explains why 3 and 9 are the next hardest digits.
```

## 5b. We *can* tell which digits are confused with 8 (same paragraph)

The test images are the same whatever the target is (checked: `uo_nn_dataset(560571,500,t,0)`
returns the same X for every t), so the true digit of each test image is known from the
labels of the ten test sets. `uo_nn_lambda_analysis.m` writes the false positives split by
true digit. For target 8, lambda = 0, GM, the 15 false positives are: 0 (3), 2 (1), 3 (2),
4 (2), 6 (2), 7 (1), 9 (4). So 11 of 15 are 0, 3, 6 or 9, which supports the overlap
explanation in item 5. (At lambda = 0.5 it is 173 of 178, see Section 3.3.)

Find:
```
Since the test labels only say target or not target, we cannot tell which digits are confused with 8. The clean
```
Replace:
```
Since the test images are the same for every target, the true digit of each test image is known from the labels of the ten test sets. For GM, 11 of the 15 false positives are 0s, 3s, 6s and 9s (3, 2, 2 and 4). The clean
```
(Apply after item 5, which introduces "The clean $7\times5$ patterns ...".)

## 6. Ordering of digits (same paragraph)

Digit 0 (train 99.2-99.4 %) sits with 2 and 6, so it was missing from the list.

Find:
```
then 2 and 6, and digits
```
Replace:
```
then 0, 2 and 6, and digits
```

## 7. Repetition in the accuracy paragraph (Section 1.4, "Across algorithms")

Also fixes "attribured". 99.30 - 98.96 = 0.34 points, i.e. 1.7 of 500 test images.

Find:
```
The differences are very small and could potentially be attribured to sampling noise. We therefore do not rank the algorithms by accuracy. Each digit was run with one seed only, so the differences cannot be separated from sampling noise with this data.
```
Replace:
```
The differences are very small: 0.34 percentage points between the best and the worst mean, or fewer than two test images per digit. Each digit was run with one seed only, so the differences cannot be separated from sampling noise with this data, and we do not rank the algorithms by accuracy.
```

## 8. Small wording fixes

- Section 1 (Problem): `pixels, henceforth the network` -> `pixels, hence the network`
- Section 1.3: `this can't be stated confidently` -> `this cannot be stated confidently`
- Title page: `October 2026` -> date of submission.

---

## 9. Wolfe count for GM on digit 8 (Section 1.2, paragraph on BFGS conditions)

Rerunning digit 8, lambda = 0, GM with the batch settings gives a strong Wolfe step in
86 of 100 iterations, not 84. The other counts (16/16, 57/61 for GM; 8/8, 18/18, 97/97
for BFGS) and the kappa(H) / cos(theta) values match. Check where 84 came from; if it was
a separate run, see the note on reproducibility below.

Find: `84/100` -> Replace: `86/100`

---

## Not text edits, but worth knowing

- **Long GM runs at lambda = 0 are not bit-reproducible.** Rerunning the batch settings
  reproduces 57 of the 60 GM/BFGS iteration counts exactly. The three that differ are the
  GM lambda = 0 runs on digits 8, 9 and 0 (e.g. digit 9 ends with ||g|| = 1.7e-3 instead of
  9.8e-4; digit 0 takes 82 iterations instead of 76). These are the badly conditioned runs
  that hit or nearly hit the iteration limit, so tiny floating-point differences grow.
  Section 1 counts digit 9 as converged with 9.8e-4 at iteration 100; that is a knife edge.

- **The line search is random.** The course function `uoBLSNW32` draws from MATLAB's
  global random generator (with `rng(1)` the first trial step is exactly `rand` = 0.417022).
  The batch is still reproducible, because `uo_nn_dataset` resets the generator right
  before each run. But the GM/BFGS iteration count changes with the generator state:
  digit 1, lambda = 0, BFGS took 12 iterations under `rng(1)` and 22 under `rng(2)`.
  Single-run differences of a few iterations between GM and BFGS are therefore partly
  noise. Section 2 quantifies this; Section 1 could mention it in one sentence.
- **Reproducibility.** The assignment says the zip must reproduce every result in the
  report. Not produced by any script in `Code/uonn` yet: the 1000-iteration GM reruns,
  the loss/gradient timings, `fig_rk_lambda0.png`, Table 4 (TPR/TNR), the SGM gradient
  norms in Table 2 (the code writes 0 for SGM), and the SWC / kappa(H) / cos(theta) counts.
  These scripts need to go in the zip. `uo_nn_lambda_analysis.m` now produces the SGM
  gradient norms, TP/FN/FP/TN, SWC counts, kappa(H) and cos(theta) for all 90 runs.
- **Files to leave out of the zip:** `uo_nn_batch_st.asv` (MATLAB autosave) and
  `firstrun.fig`. `test.m` uses `te_seed = 156057` instead of 560571, so either fix it
  or leave it out.
- **Dead code in `uosol_st.m`:** the branches for `iAC <= 3` (calls `uoBLS_st`, which is
  not in the folder), `iAC == 0` and `isd == 4` (use an undefined `h`) are never run
  with the batch settings. Harmless, but a grader running other options would hit errors.
