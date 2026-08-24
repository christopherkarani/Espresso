import ANETypes

/// What a decode step hands back to the serving-session loop.
///
/// `.selected` comes from an on-device greedy head that already resolved a token;
/// `.normalizedHidden` carries final-norm hidden state for loop-side sampling.
/// Both were produced by today's loops; the distinction preserves the fast paths.
enum LlamaDecodeProposal {
    case selected(TokenID)
    case normalizedHidden([Float])
}
