# Personalized Outreach Messages

These are ready-to-send messages for each tier 1-2 target.
All messages follow the principle: offer value first, ask small.

---

## Message 1: Stephen Panaro (GitHub issue on more-ane-transformers)

**Subject/Title**: `Espresso: 4.76x over CoreML — benchmark comparison?`

```
Hi @smpanaro,

I've been following your work on more-ane-transformers closely — especially your GPT-2-xl
results on the ANE. Really excellent engineering.

I wanted to share a related project: **Espresso** (https://github.com/christopherkarani/Espresso),
a pure-Swift inference framework that achieves 4.76x faster than CoreML via direct ANE access
using the private MIL text dialect. We've been focusing on the fused-kernel path for decode
with KV-cache, reaching ~1.9ms/tok (519 tok/s) on M3 Max.

A few things I'd be curious to get your take on:
1. How does our fused decode approach compare to your ANE pipeline?
2. Would you be open to running the Espresso benchmarks on your hardware for comparison?
3. Any architectural decisions you've found particularly important that we might be missing?

Happy to return the favor with detailed technical notes on our MIL approach — particularly
the lane-packed attention kernel design that got us past the 3x baseline.

Thanks for the great work you've been doing in this space.

— Chris (@christopherkarani)
```

---

## Message 2: Hollance (GitHub issue on neural-engine)

**Subject/Title**: `Espresso: Production ANE inference framework building on your docs`

```
Hi @hollance,

Your "Everything we know about the Apple Neural Engine" repo has been an invaluable reference —
thank you for documenting all of that. It's been cited in our internal research extensively.

I'm building **Espresso** (https://github.com/christopherkarani/Espresso), a pure-Swift
inference framework for Apple Silicon that uses the private ANE APIs you've documented to
achieve 4.76x faster inference than CoreML.

We've validated some of the MIL operator behaviors you noted (and found a few more gotchas
we'd be happy to contribute back to your docs):
- `softmax` on non-power-of-2 dimensions → `InvalidMILProgram`
- `slice_by_index` on function inputs combined with RMSNorm → compile failure workaround
- Lane-packed attention kernels for stable ANE eval across M1-M4

Would you be open to:
1. Adding a reference to Espresso in the `neural-engine` README as a production usage example?
2. A brief technical exchange about our MIL findings that might benefit your docs?

Happy to submit a PR with a findings section. This community benefits from shared knowledge.

— Chris
```

---

## Message 3: Maderix (Substack/GitHub on ANE training research)

**Subject/Title**: `Espresso + your ANE training work = complete ANE ecosystem?`

```
Hi,

I've read your Substack series "Inside the M4 Apple Neural Engine" and your work on ANE
training. Impressive research — especially the 2.8W efficiency numbers and your findings
on the convolution vs. matmul throughput differences.

I'm working on the inference side of the same problem space with **Espresso**
(https://github.com/christopherkarani/Espresso), a pure-Swift framework that achieves
519 tok/s on M3 Max for transformer inference using the same private MIL APIs.

Our experiences complement each other in an interesting way:
- You've mapped ANE training primitives
- We've mapped ANE inference kernel fusion patterns

I think there's a natural joint story here: a combined piece on "The complete ANE developer
toolkit — training and inference" that would interest the same audience following both our
work.

Would you be interested in:
1. A joint blog post comparing our benchmark methodologies and results?
2. Cross-referencing each other's technical findings?
3. Exploring whether your trained weights can be inference-optimized through Espresso?

This is a genuinely novel space and combining our research could make a strong contribution
to the community.

— Chris
```

---

## Message 4: Hugging Face (GitHub discussion on apple/coremltools or HF Discord)

**Subject/Title**: `Espresso: ANE-direct inference at 4.76x CoreML — Hub integration opportunity?`

```
Hello Hugging Face team,

I wanted to share **Espresso** (https://github.com/christopherkarani/Espresso) and explore
whether there's an integration opportunity with the Hub.

Espresso is an open-source (MIT), pure-Swift framework for Apple Silicon that achieves:
- **4.76x faster inference than CoreML** on M3 Max
- **519 tok/s** for transformer inference with KV-cache
- Zero external dependencies

The framework uses direct ANE access (private MIL text dialect) rather than CoreML's
abstraction layer, which is where the performance gain comes from.

The potential Hub integration I see:
1. An Espresso export path for transformer models (similar to CoreML export)
2. A `espresso` tag on models for Apple Silicon optimized inference
3. A model collection: `espresso-community/` with ANE-optimized checkpoints

Would this be interesting to discuss? I'm happy to start with a contribution that adds
Espresso weight export to the transformers library for community feedback.

Open to a call with the CoreML/Apple Silicon team if that would be more useful.

— Chris
```

---

## Message 5: Paul Hudson (@twostraws)

**Subject/Title**: Twitter DM or email via hackingwithswift.com contact form

```
Hi Paul,

Long-time admirer of Hacking with Swift. I've built **Espresso**
(https://github.com/christopherkarani/Espresso), a Swift framework that runs ML inference
directly on Apple's Neural Engine — 4.76x faster than CoreML.

I think there could be a compelling "Build an AI app that's actually fast" tutorial for
your audience — the kind of perf story that resonates with developers trying to ship
production ML apps.

Happy to provide technical details, example code, or be a guest for a Swift News segment.
This is genuinely new territory in the Apple Silicon space.

Would this be something you'd be interested in exploring?

Thanks,
Chris
```

---

## Key Principles for All Outreach

1. **Lead with their work**: Reference something specific they've built or written
2. **Offer value first**: Benchmark data, technical findings, cross-promotion
3. **Make the ask small**: One concrete thing — run benchmarks, add a link, chat
4. **Be specific about what Espresso achieves**: 4.76x over CoreML, 519 tok/s on M3 Max
5. **Follow up once** if no response in 2 weeks, then move on

---

## Timing

Best times to post/reach out:
- GitHub: Weekday mornings (PST)
- Twitter/X: Tuesday-Thursday, 9am-noon PST
- Discord: Same as Twitter, avoid Fridays
