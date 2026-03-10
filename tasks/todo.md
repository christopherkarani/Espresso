# TODO

- [x] Audit local artifacts and standard model paths for a real recurrent checkpoint or pretrained generation model that could drive a same-harness real-checkpoint rerun.
- [x] Verify whether the current branch already contains a principled transformer-to-recurrent conversion/export path.
- [ ] If a real recurrent artifact becomes available, rerun `scripts/reproduce_exact_4x.sh` with `INPUT_MODE=recurrent-checkpoint`, `RECURRENT_CHECKPOINT=...`, and `GENERATION_MODEL=...`.
- [ ] If no artifact exists, do not fake the rerun; document the hard blocker and define the minimum artifact/converter work needed to make the rerun honest.
- [ ] Update docs, review notes, and Wax memory with only the grounded real-checkpoint findings from this phase; hand off and flush at the checkpoint.

# Review

- The current branch contains the harness needed for a real-checkpoint rerun, but the local machine still does not have the required artifacts.
- No local `stories110M.bin` was found in the repo, the worktree, standard `assets/models` locations, or `STORIES_MODEL_PATH`, and no local `ane_stories110M_ckpt.bin` or recurrent probe-weight artifact was found either.
- The branch has no principled transformer-to-recurrent conversion path today: `GenerationWeights.load(modelPath:)` loads transformer inference weights, `Checkpoint` loads transformer training state, and `RecurrentGenerationWeightStore` only serializes already-recurrent `RWKVStyleRecurrentWeights`.
- That means a same-harness real-checkpoint rerun is blocked by artifact availability, not by benchmark plumbing.
- The honest next move is to obtain a real recurrent probe-weight artifact or implement and validate a real transformer-to-recurrent export contract before attempting to quote any real-checkpoint number.
