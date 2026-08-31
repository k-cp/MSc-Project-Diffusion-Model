# HANDOFF — end of 2026-08-31 session

Supersedes the 2026-08-28 version.

# >>> DEADLINE: 7 SEPTEMBER 2026 — SEVEN DAYS <<<

Corrected 2026-08-31; earlier versions of this file said "~30 days" and that was
wrong. **Re-plan against seven days, not thirty.** Writing is the critical path
and always was, but the slack is gone: chapters 1–3 and 5 are written, and
conclusions plus both appendices are still empty.

**The scheduling fact that follows from it:** only `sec:inp-aux` is blocked on
compute. Conclusions, Appendix A and Appendix B can all be written today with
data already in hand. Do not idle on the queue.

**`Thesis/` is committed and pushed to Overleaf** as of `fb522d4`.
The code repo is edit-only as always — nothing committed there.

**GIT POLICY CHANGED 2026-08-31.** The two repos now differ:
`Thesis/` is **auto commit + push after every edit** (re-granted; it was granted
2026-08-24 and revoked 2026-08-26, so this has flipped twice). The outer code
repo is unchanged — local edits only, never committed without being asked.
Because a `Thesis/` push publishes straight to Overleaf, the standing hazard is
sharper than before: **do not edit a file in Overleaf that is being edited here.**

---

## 1. CLUSTER: WHAT IS RUNNING RIGHT NOW

**State below is as of MIDDAY 2026-08-31 and was NOT re-confirmed.** An attempt
to check `squeue` at 15:18 failed: LRZ needs eduVPN plus TOTP 2FA and the ssh
agent had no identities, so the session could not reach the login node. **Only
you can pull live queue state.** Re-check before trusting any line below:
```
ssh di24lir@login.ai.lrz.de 'squeue -u di24lir -o "%.10i %.12j %.8T %.10M %.20R"'
```

You are capped at **10 concurrent jobs** (`QOSMaxJobsPerUserLimit`). That is the
answer to "why is nothing starting" — 10 run, the rest queue. Nothing is stuck.

**Running (10):**
* `5766945, 5766947, 5766949, 5766951, 5766953, 5766955` — `si_inpaint`, the six
  aux runs at the higher weights (gan 0.5, perc 0.5, p+g 0.5, physics 0.5,
  p+g 0.5 @32px, gan 1.0). 2000 epochs, 20–24 h each. Oldest was 14 h in at
  midday on the 31st.
* `5766957` — `fno_inpaint`, the 16 px U-Net control at 2000 epochs (~27 h).
* `5767020, 5767022, 5767024` — `fno_inpaint`, U-Net sweep at 500 epochs
  (~7 h each).

**Queued on the cap (5):** `5767026, 5767028, 5767030, 5767032, 5767034` —
the remaining U-Net sweep trainings.

**Queued on dependency (15):** every `si_inf` / `fno_inf`. The `afterok`
chaining is working; they fire on their own.

**Already finished:** `5767018` (U-Net, `single 0.6%`, 7 h) and `5767019` (its
inference, 3 min). That result is already in the CSV and the thesis.

**When more land:**
```
python eval_si_inpaint.py --csv figures/metrics_si_inpaint.csv     # on the cluster
```
then from the Mac pull **both** files — the npz is what the figures read and it
is easy to forget:
```
rsync -av di24lir@login.ai.lrz.de:Diffusion-based-Fluid-Super-resolution/figures/{metrics_si_inpaint.csv,metrics_si_inpaint_depth.npz} \
  /Users/kaya/repos/Diffusion-based-Fluid-Super-resolution/figures/
```
Then re-run `make_master_table.py`, `make_stats_table.py`, `make_sweep_table.py`,
`make_physics_figure.py`. All of them pick up new rows with no edits.

**Watch for `TIMEOUT`.** `afterok` means a walltime kill leaves a good
checkpoint whose inference never fires, and the exit code will not tell you.
`chain_si_aux.sh` prints the by-hand inference command for that case.

---

## 2. THE RESULT THAT CHANGED THE CHAPTER

The objective control now has **two** configurations and they say opposite
things:

| depth | interpolant | U-Net regressor | |
|---|---|---|---|
| 10 px (0.71 corrlen) | 0.0894 | 0.2875 | **interpolant 3.2× better** |
| 64 px (4.57 corrlen) | 0.704 | 0.694 | tie, inside the 4.2% floor |

So the tie at depth is **a depth effect, not "the formulation adds nothing"**.
The interpolant's advantage over a plain regressor on the identical architecture
is real at shallow depth and is extinguished by depth. That is the strong
version of the chapter's argument and it is now written into chapter 4.

**The convergence check matters and is already in the prose.** The 10 px U-Net
ran 500 epochs against the interpolant's 2000, so the obvious objection is
budget. The log (`fno_inpaint_5767018.log`) shows `loss_void` flat at ~0.10 for
the last 25 epochs and the learning rate annealed to 1.07e-08 — the cosine
schedule completed. It converged.

Seven more U-Net configurations are coming. If the pattern holds this becomes a
curve rather than two points.

---

## 3. WHAT CHANGED IN THE THESIS TODAY

**Late addition, `fb522d4`: depth is now defined where it is introduced.**
`03_methods.tex` defined depth bare in the mask section ("The depth of a void
pixel is its distance to the nearest observed pixel") and then silently
redefined it in the metrics list 226 lines later. The intuition — edge pixels
have observed neighbours, centre pixels do not — moved to first use, along with
the stated expectation that error grows with depth, which turns
`sec:inp-sweep`'s finding into a confirmed prediction rather than a pattern
described after the fact. The metrics passage was cut back to what it measures
and why.

**Two real errors found in the algorithms** (verified line-by-line against the
code by a workflow, then adversarially checked):
* **Mask polarity was inverted in both.** They said to zero the field *inside*
  `m`; the code and `eq:repin` both use `m` as the **keep** mask. Both now say
  `$x_0 \gets m \odot x_1$` and cite `eq:repin`.
* **"64 batches of 20" does not divide.** 64×20=1280, not 1272. It is 63
  batches of 20 plus a final 12.

**The recipe switch.** The thesis now reports the **no-xshift, no-EMA** runs
throughout, not the original nine. This moved every SI number:
* `tab:master`, `tab:stats`, `tab:sweep` and `make_depth_law_figure.py` all
  switched to `recipe="plain"`.
* `tab:sweep` was **hand-typed** and had silently kept the old values — it is
  now generated by the new `make_sweep_table.py`.
* Prose numbers that moved: the sweep spread "a factor of eighty" → **85**;
  coverage at 32 px "moves relL2 by ten per cent" → ***lowers* it by fifteen**;
  the depth law collapse 11–23 → **13–25** and 1.7–2.9 → **1.8–2.9**.
* The fractional-depth control, quoted as **98–173** and reproducible by no
  script in the repo, is now **121–154** and computed by
  `make_depth_law_figure.py` itself.
* The FNO needed a special case in both table generators — its "recipe" label is
  an artefact (the FNO runner writes no recipe suffix), so filtering by recipe
  would have emptied its column.

**A citation was wrong.** Chapter 2 said Li et al. found the learned model
"worse than POD point by point". The paper's own abstract (arXiv:2210.11921v2,
the preprint of JFM 971 A3) says *"the non-linear GAN does not outperform one of
the linear POD techniques"* — it failed to beat POD, it did not lose to it. Also
restored "extreme events", which the abstract does support and my first fix had
wrongly dropped. **The PDF was not on the Desktop and now is not either** — I
verified it online. Worth downloading; it is the one background paper you do not
have locally, and that gap is how the wrong claim survived.

**Deleted, at your call:** the depth law section, the FNO baseline section, the
classical floor section, the objective control section, sample diversity (folded
into qualitative results), the whole generalisation chapter, the Zaki precedent,
the methods chapter opener, and several smaller passages. Every cross-reference
was repaired; chapters 1–3 and 7 were edited to match. Chapter 4 went 2,737 →
1,691 words.

**Voice and simplification.** Chapters 2–4 restyled into your voice
(a workflow drafted, I fixed its overuse: "Therefore" had gone to 9 instances in
methods and every "use" had become "employ"). Then many passages simplified
further at your direction. "Stack" renamed to **"window"** throughout, matching
`fig_data_split`.

**New in the thesis:** the physics figure (`physics_in_void.pdf`, in-void
vorticity spectrum + PDF for all eight methods), the external benchmark table
(`tab_shu`), the U-Net baseline subsection in methods (it was claimed in the
introduction and reported in two tables but **described nowhere**), and
Wasserstein-1 restored alongside the KL with its 17.8% replicate floor stated.

---

## 4. NEW GENERATORS — the thesis must not disagree with the eval

`make_sweep_table.py`, `make_objective_table.py`, `make_shu_table.py`,
`make_physics_figure.py` are new. `make_master_table.py`, `make_stats_table.py`
and `make_depth_law_figure.py` were changed.

`baselines_inpaint.py` now **persists the in-void spectra** (it computed and
discarded them, exactly as `eval_si_inpaint.py` did before 2026-08-13). That is
why the classical fills appear in the physics figure at all. `make_shu_table.py`
**asserts** that all three Shu arrangements remove exactly 16,384 px, because
the section's whole argument rests on it.

`tab_objective` and `make_objective_table.py` are now **orphaned** — the
objective control section that used them was deleted. Either delete both or move
the table to Appendix B.

---

## 5. DO THESE NEXT — SEVEN-DAY PLAN

Ordering principle: **writing first, compute second.** Every remaining todo
except `sec:inp-aux` can be finished with data already on this laptop. The
queue is a bonus, not a dependency — treat every pending run as something that
may not arrive.

### Days 1–2 (Aug 31 – Sep 1): write what is not blocked
1. **Appendix A** (5 todos) — the generators, the verification scripts, the
   incident catalogue. **Criterion 2 counts software deliverables and it is
   currently empty**, so this is the highest marks-per-hour item on the list
   and it needs no new numbers.
2. **Conclusions** (3 todos). Chapter 5 already states the findings; this is
   assembly, not discovery.
3. **Appendix B** (1 todo) — or delete it, and with it the orphaned
   `tab_objective` / `make_objective_table.py` (see §4). Deciding "no appendix B"
   is a legitimate and faster answer than writing one.

### Days 3–4 (Sep 2–3): fold in whatever compute landed
4. Re-eval, pull **both** the CSV and the `.npz`, re-run the four generators,
   write `sec:inp-aux` against the reading already pre-registered in its todo
   block. **Pre-registration is the discipline that made the U-Net surprise
   usable — keep it.**
5. **HARD CUTOFF: Sep 3.** Anything not evaluated by then does not go in.
   The two-point objective control already in chapter 4 is a complete and
   defensible argument on its own; the nine-configuration column would
   strengthen it, and is not required by it. Do not let a queue slip eat the
   writing days.

### Days 5–6 (Sep 4–5): the things that will otherwise be found by an examiner
6. **`corrlen = 14 px` is still unverified and is load-bearing in three
   chapters.** It has been carried on three consecutive handoffs. Verify it or
   soften every claim that rests on it. **This is the single largest
   correctness risk in the thesis** — do not let it carry over a fourth time.
7. **The gappy POD "best rank" is selected on noise.** Rank 2 beats rank 4 by
   0.24% on relL2 against a 4.2% floor, and that arbitrary choice swings KL by
   21%. `tab:stats` prints the rank; fix the selection or disclose it.
8. **The introduction still bills the depth law as a headline contribution**
   while its section is gone (the finding survives in `sec:inp-sweep`). Either
   re-point the claim or drop it.
9. Limitations sentence on the **replicate "floors" not being seed replicates**
   (n=1, one recipe-change pair per configuration). Disclosed in
   `03_methods.tex` but not in the limitations, and **criterion 4 rewards
   exactly this kind of self-criticism.**

### Day 7 (Sep 6, leaving Sep 7 clear): full read-through
10. Compile, check every `\cref` resolves, check no `\todo` survives, confirm
    `todonotes` is switched to `[disable]` in `main.tex`. It is a **two-line
    swap, not one**: comment out the live `\usepackage[colorinlistoftodos,
    textsize=footnotesize]{todonotes}` and uncomment the
    `\usepackage[disable]{todonotes}` below it. **Miss this and every todo note
    prints in the submission PDF.**

### Not doing unless days free up
* The **centre-trained FNO** that would complete the 2×2 in
  `make_objective_table.py`. It is a few hours of compute for a table whose
  section was deleted.
* The **pod_oracle rows** (100 of them, cited nowhere). Fine to leave if
  Appendix B is dropped.

---

## 6. TRAPS, ALL HIT AT LEAST ONCE TODAY

* **`git -C <abs-path>`, always.** A bare `cd Thesis` ran against the outer repo
  once and a `git pull` went to the wrong repository.
* **Do not edit in Overleaf while asking for a rewrite here.** Three merge
  conflicts today, all from the same cause. Two resolved cleanly by luck.
  Pick one place at a time.
* **`> file` truncates before the script runs.** A failing generator emptied
  `tab_stats.tex`. Write to a temp file and `mv`.
* **The `.npz` is easy to forget.** It was two weeks stale and silently held
  pre-U-Net data; the depth-law figure was reading a different population from
  the tables for most of the day.
* Never poll `squeue`/`sacct` in a loop. ntfy topic `kaya-si-7h3k9x`.
* No local TeX toolchain — Overleaf is the only compile check.

---

## 7. UNVERIFIED / OPEN

* **`corrlen = 14 px` is still unverified** and is load-bearing in three
  chapters. Carried over from the last two handoffs.
* The **replicate "floors" are not seed replicates.** They come from one
  recipe-change pair per configuration, n=1, and the in-void correlation floor
  of 0.0% is not usable as a test. This is disclosed in `03_methods.tex` but
  worth a sentence in the limitations.
* The introduction still bills the **depth law as a headline contribution**
  while its section is gone (the finding survives in `sec:inp-sweep`).
* A **centre-trained FNO** (`sbatch run_train_fno_inpaint.sh 500 single 25 fresh
  center 4`, a few hours) would complete the 2×2 that `make_objective_table.py`
  currently cannot fill at one training position.
