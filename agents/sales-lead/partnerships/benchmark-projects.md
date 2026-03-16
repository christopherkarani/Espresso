# Apple Silicon Benchmark Project Partnerships

**Goal**: Get Espresso included in 3+ Apple Silicon benchmark suites and establish formal partnerships with 2 complementary projects.

---

## Priority Targets

### 1. ANEMLL / anemll-bench
- **Repo**: https://github.com/Anemll/anemll-bench
- **What they do**: ANE-specific benchmarking for Apple Silicon
- **Partnership Value**: Direct audience overlap — anyone comparing ANE solutions will see Espresso
- **Ask**: Add Espresso to the benchmark suite; we provide benchmark scripts + hardware numbers
- **Status**: Pending initial contact
- **Action**: Open issue titled "Add Espresso to ANE benchmark suite"

### 2. Maderix ANE (training research)
- **Repo**: https://github.com/maderix/ANE
- **What they do**: ANE training research, comprehensive private API reverse engineering
- **Partnership Value**: Cross-promotion to a highly technical audience; their training + Espresso inference = complete ANE toolkit story
- **Ask**: Technical knowledge exchange + joint content
- **Status**: Pending initial contact (same as outreach message 3)
- **Action**: Same as outreach to Maderix

### 3. Hollance / neural-engine (reference docs)
- **Repo**: https://github.com/hollance/neural-engine
- **What they do**: Definitive documentation of ANE capabilities (2.2K+ stars)
- **Partnership Value**: Being listed in the authoritative reference gives Espresso immediate credibility
- **Ask**: Mutual cross-references in READMEs
- **Status**: Pending initial contact (same as outreach message 2)
- **Action**: Same as outreach to Hollance

### 4. Stephen Panaro / more-ane-transformers
- **Repo**: https://github.com/smpanaro/more-ane-transformers
- **What they do**: ANE transformer inference, CoreML-based
- **Partnership Value**: Direct benchmark comparison; their CoreML approach vs. Espresso's direct ANE = compelling story
- **Ask**: Joint benchmark post, cross-references
- **Status**: Pending initial contact (same as outreach message 1)
- **Action**: Same as outreach to Panaro

### 5. Apple ml-ane-transformers
- **Repo**: https://github.com/apple/ml-ane-transformers
- **What they do**: Apple's official ANE transformer reference implementation
- **Partnership Value**: Being referenced by Apple is maximum credibility
- **Ask**: Add Espresso as an advanced usage example; contribute findings back to docs
- **Status**: Lower probability but high upside
- **Action**: Open discussion in GitHub repository

---

## Partnership Proposal Template

For formal partnerships, use this framework:

```
# Espresso Partnership Proposal

## What Espresso Offers
- Inference framework achieving 4.76x over CoreML
- 519 tok/s on M3 Max for transformer decode
- Pure Swift, zero dependencies, MIT licensed
- Direct ANE access via private MIL text dialect

## What We're Asking
[Specific mutual benefit — benchmark inclusion / cross-reference / joint content]

## Mutual Benefits
[What both projects gain from the partnership]

## Technical Contribution
[What Espresso will contribute to their project — code, benchmarks, documentation]
```

---

## Timeline

| Month | Action |
|-------|--------|
| March 2026 | Initial contact with all 5 targets |
| April 2026 | Follow-up with responders; begin joint work with interested partners |
| May 2026 | First joint content published (target: 2+ pieces) |
| June 2026 | WWDC timing — publish benchmark comparisons from partnerships |
