# Conference Talk Proposals

## Target Conferences

| Conference | Dates | Location | CFP Status | Priority |
|-----------|-------|----------|-----------|----------|
| Deep Dish Swift 2026 | April 12-14 | Chicago | Check site | HIGH |
| try! Swift Tokyo 2026 | Spring 2026 | Tokyo | https://cfp.tryswift.jp/ | HIGH |
| WWDC 2026 Labs | June 2026 | Cupertino | Request-based | MEDIUM |
| SwiftConf 2026 | September 2026 | Cologne | Watch site | MEDIUM |
| iOS Conf SG 2026 | TBD | Singapore | Watch site | LOW |

---

## Proposal 1: Full Technical Talk (40 min)

**Title**: "4.76x Faster Than CoreML: Building High-Performance ML on Apple's Neural Engine"

**Abstract**:
Apple's Neural Engine can deliver extraordinary inference performance — but CoreML's abstraction layer leaves significant throughput on the table. In this talk, we'll explore how Espresso, a pure-Swift open-source framework, achieves 4.76x faster transformer inference than CoreML by directly targeting the ANE's MIL text dialect.

We'll cover:
- The architecture of Apple's Neural Engine and where CoreML loses performance
- How the MIL (Model Intermediate Language) text format enables kernel fusion and direct dispatch
- Designing fused attention + FFN kernels that reduce per-layer dispatch overhead by 6x
- The lane-packed attention approach that avoids ANE eval instability across M1-M4
- Reaching 519 tokens/second for transformer decode on M3 Max

You'll leave with a deep understanding of Apple Silicon's ML pipeline, practical techniques for ANE optimization, and working knowledge of Espresso's architecture that you can apply to your own ML projects.

**Target Audience**: iOS/macOS developers with interest in performance, ML engineers targeting Apple hardware
**Duration**: 40 minutes + Q&A
**Demo**: Live benchmark comparison: CoreML vs. Espresso on M-series hardware

---

## Proposal 2: Quick Talk (20 min)

**Title**: "Unlock Your Mac's Neural Engine: From CoreML to Direct ANE Access"

**Abstract**:
What if your on-device ML app could run 4.76x faster with a single framework swap? Apple's Neural Engine is dramatically underutilized by standard CoreML deployments. This talk introduces Espresso, a pure-Swift framework for direct ANE inference, and shows you when and how to use it.

We'll cover the key decision: when CoreML is the right choice vs. when direct ANE access is worth it, and walk through integrating Espresso into a real Swift project for maximum inference performance.

**Duration**: 20 minutes
**Demo**: Side-by-side CoreML vs. Espresso token generation

---

## Proposal 3: Lightning Talk (10 min)

**Title**: "519 Tokens Per Second on Your MacBook: A 3-Minute Espresso Demo"

**Abstract**:
A live demo of Espresso generating text at 519 tokens/second on M3 Max, followed by a brief explanation of how it works. No CoreML. No Python. Pure Swift, direct ANE access, 4.76x faster than the standard path.

**Duration**: 10 minutes
**Best for**: try! Swift quick talks, conference showcase slots

---

## WWDC Labs Strategy

For WWDC 2026, request **one-on-one ANE optimization labs** with:
1. Core ML team — discuss Espresso's findings; ask if any of our MIL optimizations can become public API
2. Performance tools team — request guidance on profiling private ANE dispatch timing
3. Machine Learning Infrastructure team — discuss potential for direct ANE API exposure

**Framing**: "We've built an open-source framework that uses private ANE APIs and achieved 4.76x over CoreML. We'd love to understand the roadmap for official API exposure and how we can contribute to the ecosystem."

---

## Submission Checklist

For each conference submission:
- [ ] Research CFP deadline (typical: 3-4 months before conference)
- [ ] Tailor abstract to conference audience (e.g., more beginner-friendly for try! Swift Tokyo)
- [ ] Prepare speaker bio emphasizing Espresso and ANE expertise
- [ ] Include benchmark data/visuals in submission
- [ ] Mention live demo availability
- [ ] Request "performance" or "machine learning" track placement

---

## Speaker Bio

```
Chris Karani is the creator of Espresso, a pure-Swift framework for direct Apple Neural Engine
inference that achieves 4.76x faster performance than CoreML on Apple Silicon. He specializes
in low-level performance optimization on Apple hardware and has spent the last year reverse-
engineering the ANE's private MIL text dialect to build production-ready inference infrastructure.
```
