## Summary

<!-- What does this PR do? 1–3 sentences. -->

## Changes

-
-

## Testing

- [ ] Unit tests pass (`swift test`)
- [ ] Hardware tests pass (`ANE_HARDWARE_TESTS=1 swift test`) — skip if no ANE
- [ ] New code has test coverage

## Benchmark impact

<!-- If this touches the hot path, include a before/after. Use ./scripts/reproduce_local_real_artifact_claim.sh -->

None / N/A

## Checklist

- [ ] Code compiles under Swift 6.2 strict concurrency (`.swiftLanguageMode(.v6)`)
- [ ] No external dependencies added
- [ ] Files ≤400 lines, functions ≤50 lines
- [ ] Commit messages follow `type: summary` convention
