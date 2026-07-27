#!/bin/bash
# ---------------------------------------------------------------------------
# Submit the whole baseline + DPS experiment sweep in one go.
#
#   ./submit_sweep.sh            # submit everything
#   DRY=1 ./submit_sweep.sh      # print what WOULD be submitted, submit nothing
#   ONLY=dps ./submit_sweep.sh   # just one group: physics | rsweep | jcp | tsweep | dps
#
# WHY THE PER-JOB --time OVERRIDES: run_inference_conditional.sh asks for 4 h,
# but a normal r=20 run takes ~20 min. A too-large request is what makes Slurm
# park a job behind a maintenance reservation ("ReqNodeNotAvail"), because it
# will not start anything that cannot finish before the window opens. The times
# below are ~2x measured runtime -- enough headroom, small enough to slip into
# a gap. Scale them all with TIMESCALE if the queue disagrees.
#
# Maintenance 2026-07-28 06:00-20:00 BST: anything still queued then simply
# starts after 20:00. Nothing is lost, it just waits.
# ---------------------------------------------------------------------------

DRY="${DRY:-0}"
ONLY="${ONLY:-all}"
S=run_inference_conditional.sh

submitted=0

sub() {   # sub <walltime> <description> <sbatch args...>
    local t="$1"; shift
    local desc="$1"; shift
    if [ "$DRY" = "1" ]; then
        printf '  [dry] %-40s (--time=%s)  %s\n' "$desc" "$t" "$*"
        return
    fi
    local jid
    if jid=$(sbatch --time="$t" --parsable "$@"); then
        printf '  %-10s %-40s (--time=%s)\n' "$jid" "$desc" "$t"
        submitted=$((submitted + 1))
    else
        # NOTE: "Socket timed out on send/recv operation" means sbatch never got
        # the controller's REPLY -- the job is usually created anyway. Check with
        # `squeue --me` before resubmitting, or you end up with two identical jobs
        # writing to the same output folder and clobbering each other.
        printf '  FAILED     %-40s  <- check squeue before resubmitting!\n' "$desc"
    fi
    # Rapid-fire submissions are what provoke those controller timeouts.
    sleep 2
}

want() { [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ]; }

# --- 0. baseline retraining -- OPT-IN ONLY -----------------------------------
# Deliberately NOT part of "all": this is a ~12 h job and you never want two of
# them running at once, which is exactly what would happen if you re-ran this
# script to top up the inference sweep. Submit it explicitly:
#     ONLY=train ./submit_sweep.sh
# The script itself asks for 24 h, which cannot fit before a maintenance window;
# override with TRAINTIME if there is a usable gap (resume makes this low-risk:
# if it is killed, `sbatch run_train_baseline.sh resume` picks up from ckpt.pth).
#     TRAINTIME=13:00:00 ONLY=train ./submit_sweep.sh
if [ "$ONLY" = "train" ]; then
    echo "baseline retraining (~12 h; produces conditional_ckpt_mine.pth):"
    sub "${TRAINTIME:-24:00:00}" "train conditional baseline" run_train_baseline.sh
    echo
    echo "When it finishes, publish the weights:"
    echo "  cp train_ddpm/experiments/km256_mine/logs/weights/km256_mine/ckpt.pth \\"
    echo "     pretrained_weights/conditional_ckpt_mine.pth"
    echo "then:  sbatch run_inference_conditional.sh baseline mine"
    exit 0
fi

# --- 1. physics ablation: which half of Shu et al.'s guidance does the work ---
# 'both' (the repo default) you already have as guided_recons_..._w0.0
if want physics; then
    echo "physics ablation (vs your existing default run):"
    sub 00:45:00 "cond only   -> _physcond"    $S baseline given 1 0 cond
    sub 00:45:00 "linear only -> _physlinear"  $S baseline given 1 0 linear
    sub 00:45:00 "no physics  -> _physnone"    $S baseline given 1 0 none
fi

# --- 2. reverse-step sweep: closes the integration-resolution gap with SI ----
# SI uses 100 steps / 200 net calls; the baseline uses 20 / 40.
if want rsweep; then
    echo "reverse-step sweep (cost scales with r):"
    sub 01:30:00 "r=50        -> _t400_r50"    --export=ALL,R=50  $S baseline
    sub 02:30:00 "r=100       -> _t400_r100"   --export=ALL,R=100 $S baseline
fi

# --- 3. the two JCP mechanisms the shipped config disables -------------------
if want jcp; then
    echo "JCP mechanisms (iterative refinement / classifier-free guidance):"
    sub 01:30:00 "sample_step=3        -> _ss3"      $S baseline given 3
    sub 01:30:00 "sample_step=5        -> _ss5"      $S baseline given 5
    sub 00:45:00 "w=3.0                -> _w3.0"     $S baseline given 1 3.0
    sub 01:30:00 "w=3.0 + ss=3     -> _w3.0_ss3"     $S baseline given 3 3.0
fi

# --- 4. noise-level sweep: how much of the measurement to destroy ------------
if want tsweep; then
    echo "noise-level sweep (t=400 is your existing run):"
    sub 00:45:00 "t=300       -> _t300_r20"    --export=ALL,T=300 $S baseline
    sub 00:45:00 "t=200       -> _t200_r20"    --export=ALL,T=200 $S baseline
fi

# --- 5. DPS + physics: the unexplored cell (measurement AND physics) ---------
# 'x0hat' adds a second backward pass through the UNet, hence the longer slot.
if want dps; then
    echo "DPS + physics:"
    sub 01:30:00 "cond only        -> _z3.0_cond"          $S dps 3.0 none 0 cond
    sub 02:00:00 "force only       -> _physx0hat_lam1.0"   $S dps 3.0 x0hat 1.0
    sub 02:00:00 "cond + force     -> _cond_physx0hat"     $S dps 3.0 x0hat 1.0 cond
fi

echo
if [ "$DRY" = "1" ]; then
    echo "DRY RUN -- nothing submitted. Re-run without DRY=1 to submit."
else
    echo "$submitted job(s) submitted.  Watch with:  squeue --me"
    echo "Start estimates (useful around maintenance):  squeue --me --start"
fi
