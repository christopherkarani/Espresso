# ObjC Golden Fixtures (Phase 6b)

These files are required by ANERuntime cross-validation tests when
`OBJC_CROSS_VALIDATION=1` is enabled.

Expected files (raw little-endian `Float32` vectors):
- `fwd_attn_oOut_seq256_f32le.bin`
- `fwd_ffn_y_seq256_f32le.bin`
- `ffn_bwd_dx_seq256_f32le.bin`

Each fixture stores exactly `ModelConfig.dim * ModelConfig.seqLen` float values.

If fixtures are missing, cross-validation tests skip with an explicit message.
