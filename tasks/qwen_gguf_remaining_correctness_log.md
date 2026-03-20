# Qwen GGUF Remaining Correctness Log

## 2026-03-20

### Experiment 1: Establish current boundary and comparison harness path
- Hypothesis: the remaining late-token mismatch can be narrowed by directly comparing raw GGUF dequantized tensors against the fresh artifact `.float32.bin` sidecars, using repo-native loader/dequantization code.
- Exact commands run:
```bash
ls -1 /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367
python3 - <<'PY'
from pathlib import Path
root = Path('/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367')
for rel in ['layers/27/w2.float32.bin','layers/27/wq.float32.bin','layers/27/wk.float32.bin','layers/27/wv.float32.bin']:
    p = root / rel
    print(rel, p.exists(), p.stat().st_size if p.exists() else None)
PY
sed -n '1,260p' /Users/chriskarani/CodingProjects/Edgerunner/Sources/EspressoEdgeRunner/WeightConverter.swift
sed -n '1,240p' /Users/chriskarani/CodingProjects/Edgerunner/Sources/EspressoEdgeRunner/EspressoTensorNameMapper.swift
sed -n '1,260p' /Users/chriskarani/CodingProjects/Edgerunner/Sources/EdgeRunnerIO/GGUF/GGUFLoader.swift
sed -n '1,260p' /Users/chriskarani/CodingProjects/Edgerunner/Sources/EspressoEdgeRunner/DequantDispatcher.swift
```
- Exact result:
  - Fresh artifact `espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367` exists and contains full float32 sidecars for the required layer-27 tensors.
  - Sidecar sizes confirm expected tensor families are present:
    - `layers/27/w2.float32.bin` = `12582912` bytes
    - `layers/27/wq.float32.bin` = `8388608` bytes
    - `layers/27/wk.float32.bin` = `4194304` bytes
    - `layers/27/wv.float32.bin` = `4194304` bytes
  - Converter code confirms the current production path writes Qwen sidecars after dequantization and any converter permutation decisions in `WeightConverter`.
- Whether token behavior changed:
  - No. This was a read-only prerequisite and harness setup step.
- What invariant was confirmed or ruled out:
  - Confirmed that the comparison boundary is available on disk for the exact tensors requested by the task.
  - Confirmed that the sidecars record post-converter semantics, so direct raw-GGUF-vs-sidecar comparison can isolate converter invariants without involving the ANE runtime.
- Whether the path is now a dead end:
  - No. This is the active narrowing path.

### Experiment 2: Compare the first requested late-token tensor set on a fresh kept artifact
- Hypothesis: the remaining late-prefix mismatch is still caused by a converter semantic error in the first requested late-token-relevant tensor family (`blk.27.ffn_down.weight`, `blk.27.attn_q.weight`, `blk.27.attn_k.weight`, `blk.27.attn_v.weight`).
- Exact commands run:
```bash
ESPRESSO_GGUF_KEEP_WEIGHT_DIR=1 ./.build/debug/EspressoGGUFRunner \
  /tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
  /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  Hello 1

ESPRESSO_DEBUG_COMPARE_RAW_GGUF_SIDECARS=1 \
ESPRESSO_DEBUG_GGUF_MODEL=/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
ESPRESSO_DEBUG_WEIGHT_DIR=/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_17334918-3275-4FCF-AB7E-D79F5292E778 \
ESPRESSO_DEBUG_TENSOR_NAMES='blk.27.ffn_down.weight,blk.27.attn_q.weight,blk.27.attn_k.weight,blk.27.attn_v.weight' \
swift test --filter debugCompareRawGGUFTensorsAgainstArtifactSidecars
```
- Exact result:
  - Fresh kept artifact: `/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_17334918-3275-4FCF-AB7E-D79F5292E778`
  - Fresh cold-start on that artifact still succeeds and prints `Hello Answer`.
  - Direct raw-GGUF-vs-sidecar compare on the first requested tensor set is exact:
    - `blk.27.ffn_down.weight`: shape `[3072, 1024]`, count `3145728`, max abs diff `0`, mean abs diff `0`, cosine `1`
    - `blk.27.attn_q.weight`: shape `[1024, 2048]`, count `2097152`, max abs diff `0`, mean abs diff `0`, cosine `1`
    - `blk.27.attn_k.weight`: shape `[1024, 1024]`, count `1048576`, max abs diff `0`, mean abs diff `0`, cosine `1`
    - `blk.27.attn_v.weight`: shape `[1024, 1024]`, count `1048576`, max abs diff `0`, mean abs diff `0`, cosine `1`
- Whether token behavior changed:
  - No. This was a comparison-only check.
- What invariant was confirmed or ruled out:
  - Confirmed the fresh artifact preserves the exact raw GGUF semantics for the first requested late-token tensor family.
  - Ruled out transpose / QK reorder / V reorder explanations for those mapped tensors unless env leakage is separately proven.
- Whether the path is now a dead end:
  - Yes for the first requested converter tensor family. The remaining bug is not in those converted weights.

### Experiment 3: Exhaust the mapped converter surface before touching runtime again
- Hypothesis: a materially mismatching mapped tensor still exists somewhere else in the converted Qwen artifact.
- Exact commands run:
```bash
python3 - <<'PY'
import json
path='/tmp/qwen_compare_all_mapped_XXXXXX.json'
with open(path) as f:
    reports=json.load(f)
nonzero=[r for r in reports if r['direct']['maxAbsDiff']!=0 or r['direct']['meanAbsDiff']!=0 or r['direct']['cosineSimilarity']!=1]
print('REPORT_COUNT', len(reports))
print('NONZERO_COUNT', len(nonzero))
PY
```
- Exact result:
  - `REPORT_COUNT 310`
  - `NONZERO_COUNT 0`
  - The all-mapped raw-GGUF-vs-sidecar compare in the sibling `Edgerunner` repo found zero materially mismatching mapped tensors.
- Whether token behavior changed:
  - No. This was a summarization step over the comparison output.
- What invariant was confirmed or ruled out:
  - Confirmed the converter is semantically correct for all 310 mapped tensors on the current branch.
  - Promoted converter-side transpose / row-order / mapped-source-selection theories to dead ends for the mapped tensor surface.
- Whether the path is now a dead end:
  - Yes. Converter semantics are no longer the active boundary.

### Experiment 4: Prove the remaining mismatch is before the final head, not in the converted LM head
- Hypothesis: the late-prefix mismatch still comes from a converter-side top-weight or tied-head semantic error.
- Exact commands run:
```bash
python3 - <<'PY'
import json, subprocess
from tokenizers import Tokenizer
model='/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf'
prompt="Hello Answer, I'm sorry for the"
tok=Tokenizer.from_file('/Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068/tokenizer.json')
out = subprocess.run([
    '/opt/homebrew/bin/llama-cli','-m',model,'-p',prompt,'-n','1','--temp','0','--top-k','1','--top-p','1','--min-p','0','--no-display-prompt','--no-warmup','--simple-io','-no-cnv','--threads','4','--ctx-size','512'
], check=True, capture_output=True, text=True).stdout.rstrip('\n')
print(json.dumps({'text': out, 'generated_tokens': tok.encode(out).ids}, ensure_ascii=False, indent=2))
PY

ANE_HARDWARE_TESTS=1 \
ESPRESSO_DEBUG_WEIGHT_DIR=/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367 \
ESPRESSO_DEBUG_PROMPT_TOKENS='9707,21806,11,358,2776,14589,369,279' \
swift test --filter test_debugQwenRawPromptNextTokenFromWeightDir

ANE_HARDWARE_TESTS=1 \
ESPRESSO_DEBUG_WEIGHT_DIR=/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367 \
ESPRESSO_DEBUG_GGUF_MODEL=/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
ESPRESSO_DEBUG_PROMPT_TOKENS='9707,21806,11,358,2776,14589,369,279' \
ESPRESSO_DEBUG_CHECK_RAW_GGUF_TOP_WEIGHTS_NEXT_TOKEN=1 \
swift test --filter test_debugQwenRawGGUFTopWeightsPromptNextTokenFromWeightDir
```
- Exact result:
  - Raw GGUF late-prefix next token: text `" previous"`, token `[3681]`.
  - Existing kept artifact late-prefix next token: token `21340`.
  - Raw GGUF top weights applied directly to Espresso’s final hidden state still return `21340`, exactly matching the artifact LM head on that hidden state.
- Whether token behavior changed:
  - No. This localized the mismatch boundary.
- What invariant was confirmed or ruled out:
  - Confirmed the remaining bug was upstream of the final RMS/LM-head path.
  - Ruled out tied-weight / output-weight converter semantics as the remaining explanation.
- Whether the path is now a dead end:
  - Yes. Final-head converter semantics are a dead end.

### Experiment 5: Remove repeated FP16 activation snapping from the exact CPU decode path
- Hypothesis: the exact CPU decode path is not actually exact for Qwen because it repeatedly rounds Q/K/V, RoPE outputs, residual projections, and FFN outputs back to FP16 between layers.
- Exact commands run:
```bash
swift test --filter QwenGGUFSidecarComparisonTests
swift build --product espresso-generate

ESPRESSO_DEBUG_CPU_EXACT_DECODE=1 ./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367 \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 1 "Hello Answer, I'm sorry for the"

ESPRESSO_DEBUG_CPU_EXACT_DECODE=1 \
ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES=1 \
./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367 \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 1 "Hello Answer, I'm sorry for the"

ESPRESSO_DEBUG_CPU_EXACT_DECODE=1 \
ESPRESSO_DEBUG_CPU_EXACT_DECODE_KEEP_FP32_INTERMEDIATES=1 \
./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_B2601A85-B29B-4983-BCFC-5C9A99DCA367 \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 8 "Hello"
```
- Exact result:
  - Focused tests for the new exact-CPU precision helper pass.
  - Default exact CPU decode on the existing kept artifact still emits late-prefix token `21340`.
  - Keeping exact-CPU intermediates in FP32 flips the late-prefix token to `3681`.
  - Keeping exact-CPU intermediates in FP32 also produces the full `Hello` 8-token continuation:
    - `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
    - text `Hello Answer, I'm sorry for the previous`
- Whether token behavior changed:
  - Yes. The late-prefix token changed from `21340` to `3681`, and the full `Hello` continuation matched raw GGUF.
- What invariant was confirmed or ruled out:
  - Confirmed the runtime root cause: repeated FP16 intermediate rounding inside `generateIncrementalExactCPULlama`.
  - Ruled out the remaining converter invariants as the active cause of the late-token mismatch.
- Whether the path is now a dead end:
  - Yes for converter-first narrowing. The active fix moved into Espresso runtime precision.

### Experiment 6: Promote the exact-CPU FP32 behavior to the default and verify on a fresh artifact
- Hypothesis: making exact CPU decode keep FP32 intermediates by default fixes branch correctness without regressing fresh cold-start.
- Exact commands run:
```bash
swift test --filter QwenGGUFSidecarComparisonTests
swift build --product EspressoGGUFRunner

ESPRESSO_GGUF_KEEP_WEIGHT_DIR=1 ./.build/debug/EspressoGGUFRunner \
  /tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf \
  /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  Hello 1

./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_17334918-3275-4FCF-AB7E-D79F5292E778 \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 1 "Hello Answer, I'm sorry for the"

./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_17334918-3275-4FCF-AB7E-D79F5292E778 \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 8 "Hello"

python3 - <<'PY'
import json, subprocess
from tokenizers import Tokenizer
model='/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf'
prompt='Hello'
tok=Tokenizer.from_file('/Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068/tokenizer.json')
out = subprocess.run([
    '/opt/homebrew/bin/llama-cli','-m',model,'-p',prompt,'-n','8','--temp','0','--top-k','1','--top-p','1','--min-p','0','--no-display-prompt','--no-warmup','--simple-io','-no-cnv','--threads','4','--ctx-size','512'
], check=True, capture_output=True, text=True).stdout.rstrip('\n')
print(json.dumps({'text': out, 'generated_tokens': tok.encode(out).ids}, ensure_ascii=False, indent=2))
PY
```
- Exact result:
  - Focused tests still pass after promoting FP32 intermediates to the default exact-CPU behavior.
  - Fresh kept artifact: `/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_17334918-3275-4FCF-AB7E-D79F5292E778`
  - Fresh cold-start still succeeds and prints `Hello Answer`.
  - Fresh late-prefix next token now matches raw GGUF: `[9707, 21806, 11, 358, 2776, 14589, 369, 279] -> 3681`
  - Fresh full `Hello` 8-token continuation now matches raw GGUF exactly:
    - Espresso: `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
    - raw GGUF: `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
    - text: `Hello Answer, I'm sorry for the previous`
- Whether token behavior changed:
  - Yes. The branch is now correct on the fresh artifact without any precision override env var.
- What invariant was confirmed or ruled out:
  - Confirmed the precise root cause and production fix: exact CPU Qwen decode must keep intermediates in FP32.
  - Confirmed the fresh artifact still preserves exact raw GGUF tensor semantics on the requested first comparison set.
- Whether the path is now a dead end:
  - No. This is the final fixed branch state.

### Experiment 7: Add permanent 0.6B regressions and gate stale Qwen experiment-only paths
- Hypothesis: the 0.6B fix should be locked down with permanent regressions, while stale one-off Qwen probes should stay available only behind an explicit experiment gate.
- Exact commands run:
```bash
rg -n "qwen060B|qwenManual|ESPRESSO_ENABLE_QWEN_EXPERIMENT_TESTS|generateTokensExactCPUForTesting" \
  Tests/RealModelInferenceTests/QwenGGUFRegressionSupport.swift \
  Tests/RealModelInferenceTests/QwenGGUFRegressionTests.swift \
  Tests/RealModelInferenceTests/QwenGGUFSidecarComparisonTests.swift \
  Tests/RealModelInferenceTests/RealModelInferenceTests.swift \
  Sources/RealModelInference/RealModelInferenceEngine.swift

swift test --filter 'QwenGGUF(Regression|SidecarComparison)Tests'
```
- Exact result:
  - Permanent 0.6B regressions now exist for:
    - fresh-artifact `Hello` 8-token parity
    - fresh-artifact late-prefix next-token parity
    - exact-CPU late-prefix parity with FP32 intermediates by default
  - The exact-CPU token-sequence helper is covered by a focused negative-path test (`maxTokens > 0`).
  - Legacy Qwen experiment probes in `RealModelInferenceTests.swift` are now hard-gated behind `ESPRESSO_ENABLE_QWEN_EXPERIMENT_TESTS=1`.
  - The combined focused suite passed:
    - `16 tests in 2 suites passed after 552.416 seconds`
    - `Hello` token `0` stayed `21806`
    - late-prefix token stayed `3681`
    - full `Hello` continuation stayed `[21806, 11, 358, 2776, 14589, 369, 279, 3681]`
- Whether token behavior changed:
  - No. This locked in the already-correct branch behavior and removed stale always-on experiment noise.
- What invariant was confirmed or ruled out:
  - Confirmed the 0.6B fix is now protected by permanent regression coverage.
  - Confirmed the remaining useful long-term correctness oracles are the raw-vs-sidecar compare, tied-head selection, exact-CPU helper guards, and env-gated manual larger-model spot checks.
- Whether the path is now a dead end:
  - Yes for ungated stale experiment probes. They should not run by default again.

### Experiment 8: Restore the missing protected 1.7B model and derive grounded raw-GGUF larger-model oracles
- Hypothesis: broader 1.7B and 4B spot checks need raw GGUF token oracles first; if the protected 1.7B model is missing locally, it must be restored before any grounded comparison.
- Exact commands run:
```bash
curl -L --fail --progress-bar -C - \
  -o /tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf.part \
  https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf \
  && mv /tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf.part /tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf

ls -lh /tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf

python3 - <<'PY'
import json, subprocess
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068/tokenizer.json')
for model in [
    '/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf',
    '/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf',
]:
    for prompt, n in [("Hello Answer, I'm sorry for the", 1), ("Hello", 8)]:
        out = subprocess.run([
            '/opt/homebrew/bin/llama-cli', '-m', model, '-p', prompt, '-n', str(n),
            '--temp', '0', '--top-k', '1', '--top-p', '1', '--min-p', '0',
            '--no-display-prompt', '--no-warmup', '--simple-io', '-no-cnv',
            '--threads', '4', '--ctx-size', '512'
        ], check=True, capture_output=True, text=True).stdout.rstrip('\\n')
        print(json.dumps({
            'model': model,
            'prompt': prompt,
            'generated_text': out,
            'generated_tokens': tok.encode(out).ids,
        }, ensure_ascii=False))
PY
```
- Exact result:
  - The protected `Qwen3-1.7B-Q8_0.gguf` model was restored successfully at `/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf` with size `1834426016` bytes.
  - Raw GGUF 1.7B oracle:
    - late-prefix generated text `" confusion"`, token `[21340]`
    - `Hello` 8-token continuation generated text `": I'm trying to understand the concept"`, tokens `[25, 358, 2776, 4460, 311, 3535, 279, 7286]`
  - Raw GGUF 4B oracle:
    - late-prefix generated text `" inconvenience"`, token `[60009]`
    - `Hello` 8-token continuation generated text `"<3\nHello<3\nHello"`, tokens `[27, 18, 198, 9707, 27, 18, 198, 9707]`
- Whether token behavior changed:
  - No for Espresso. This established grounded larger-model goldens before further spot checks.
- What invariant was confirmed or ruled out:
  - Confirmed the larger models do not share the 0.6B golden tokens, so permanent 1.7B/4B tests must stay expectation-driven, not hard-coded to 0.6B outputs.
  - Confirmed the raw-GGUF late-prefix and `Hello` goldens needed for larger-model spot checks.
- Whether the path is now a dead end:
  - No. These are the active larger-model reference oracles.

### Experiment 9: Broaden the larger-model spot checks on 1.7B with exact-CPU parity and fresh-artifact default-path evidence
- Hypothesis: the 1.7B model should match the raw GGUF late-prefix and `Hello` continuation under the new exact-CPU oracle, and the prepared-artifact default path should either produce the same token or expose a concrete runtime blocker.
- Exact commands run:
```bash
ESPRESSO_QWEN_MANUAL_MODEL_PATH=/tmp/edgerunner-models/Qwen3-1.7B-Q8_0.gguf \
ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN=21340 \
ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS=25,358,2776,4460,311,3535,279,7286 \
swift test --filter qwenManual

./.build/debug/espresso-generate generate \
  --weights /var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_29A6FDF4-6CD3-4BD2-81C7-DBE431D1B81E \
  --tokenizer /Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --no-bootstrap --json -n 1 "Hello Answer, I'm sorry for the"
```
- Exact result:
  - The exact-CPU manual 1.7B spot checks both passed on the fresh prepared artifact:
    - `Manual Qwen exact CPU late-prefix token matches expected` passed after `858.530` seconds.
    - `Manual Qwen Hello continuation matches expected tokens` passed after `284.137` seconds.
    - Combined suite: `2 tests in 1 suite passed after 1142.670 seconds`.
  - The fresh artifact used by that run was `/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_29A6FDF4-6CD3-4BD2-81C7-DBE431D1B81E`.
  - Reusing that same artifact on the default `espresso-generate` path did not emit a token or JSON result before manual termination.
  - Instead it entered a sustained ANE compile retry loop with repeated lines of:
    - `ANE compile retrying (2/5) after transient compiler failure`
    - `ANE compile retrying (3/5) after transient compiler failure`
    - `ANE compile retrying (4/5) after transient compiler failure`
    - `ANE compile retrying (5/5) after transient compiler failure`
  - The `espresso-generate` process was still consuming `~100%` CPU after `5m56s` on a single-token late-prefix request, so it was manually terminated without a token result to avoid burning the remaining disk/CPU budget.
- Whether token behavior changed:
  - Yes for 1.7B exact-CPU parity: the manual checks proved Espresso matches raw GGUF on both late-prefix and `Hello` continuation for that model.
  - No successful token was produced on the default fresh-artifact path because runtime compilation did not converge.
- What invariant was confirmed or ruled out:
  - Confirmed the current exact-CPU Qwen path generalizes beyond 0.6B to 1.7B for both target prompts.
  - Confirmed the larger-model blocker is now on the fresh-artifact default runtime path, not in raw GGUF semantics or the exact-CPU oracle.
- Whether the path is now a dead end:
  - No. This is active blocker evidence for the larger-model merge gate.

### Experiment 10: Reclaim disk only from disposable temp conversion directories
- Hypothesis: larger-model verification can proceed safely if only disposable `/var/folders/.../T/espresso_gguf_*` directories are removed, while protected source models and `.artifacts` are preserved.
- Exact commands run:
```bash
python3 -c "from pathlib import Path; import shutil; root=Path('/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T'); removed=0
for path in root.glob('espresso_gguf_*'):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False); removed+=1
print(f'removed={removed}')"

df -h /System/Volumes/Data
```
- Exact result:
  - Cleanup was limited to disposable temp conversion directories under `/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_*`.
  - Earlier cleanup rounds removed `8` temp directories and restored roughly `82 GiB` free, then removed `5` temp directories and restored roughly `60 GiB` free when later large-model prep re-filled the disk.
  - No protected source models were deleted.
  - No `.artifacts` directories were deleted.
- Whether token behavior changed:
  - No. This was an infrastructure-only recovery step.
- What invariant was confirmed or ruled out:
  - Confirmed the allowed cleanup boundary is sufficient to recover verification capacity without touching protected assets.
- Whether the path is now a dead end:
  - No. This is the approved recovery path whenever larger-model temp artifacts exhaust local disk.

### Experiment 11: Push the 4B exact-CPU spot check until the current verification budget breaks
- Hypothesis: the 4B model should either match the raw GGUF late-prefix and `Hello` continuation under the new exact-CPU oracle, or expose a concrete runtime/storage limit for the current full-sidecar verification flow.
- Exact commands run:
```bash
ESPRESSO_QWEN_MANUAL_MODEL_PATH=/tmp/edgerunner-models/Qwen3-4B-Q8_0.gguf \
ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN=60009 \
ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS=27,18,198,9707,27,18,198,9707 \
swift test --filter qwenManual
```
- Exact result:
  - The run prepared a fresh full-sidecar 4B artifact at:
    - `/var/folders/ns/xmz0zmpj7p148vdgr4bwzp8h0000gn/T/espresso_gguf_DC499409-5DBF-492A-892B-23319CB25E6D`
  - That artifact grew to `23018.61 MiB` on disk.
  - The `swiftpm-testing-helper` process was still active after `32m53s`, consuming `~100%` CPU and `33.3%` memory, with no completed test output and no token result emitted yet.
  - Remaining free space at that point was down to `17 GiB`.
  - The run was manually terminated to avoid burning the remaining local budget without producing a new semantic result.
  - Grounded context from earlier steps in this session still stands:
    - raw GGUF 4B late-prefix token is `[60009]`
    - raw GGUF 4B `Hello` continuation is `[27, 18, 198, 9707, 27, 18, 198, 9707]`
    - a prior exact-CPU late-prefix probe on 4B already returned `60009`, matching the raw GGUF late-prefix oracle
- Whether token behavior changed:
  - No new token result was produced before termination.
- What invariant was confirmed or ruled out:
  - Confirmed the remaining unresolved larger-model item is not the 4B late-prefix semantic target itself.
  - Confirmed the practical blocker is the current cost of full-sidecar 4B exact-CPU `Hello` verification on this machine and branch, not a newly proven 4B semantic mismatch.
- Whether the path is now a dead end:
  - No. This is the current larger-model runtime-budget blocker.
