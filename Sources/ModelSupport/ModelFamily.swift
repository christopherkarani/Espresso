import Foundation

/// Centralized model-family recognition for runtime policy defaults.
///
/// Stories 110M is the retained public demo / benchmark artifact. Several ANE
/// serving defaults (hybrid cached bindings, fused exact head, ANE classifier
/// allowlist) currently prefer this family. Qwen 1.5B also defaults hybrid
/// cached bindings on. Broader Llama policy is still opt-in. Keep every
/// name-based check here so special cases do not scatter.
public enum ModelFamily: Sendable {
    /// Stable substring / canonical id used for Stories 110M artifacts.
    public static let stories110mToken = "stories110m"

    /// Returns true when `name` identifies a Stories 110M (or derivative) artifact.
    ///
    /// Matches:
    /// - `stories110m`
    /// - `stories110m-ctx256`
    /// - `llama2.c-stories110M` (case-insensitive)
    public static func isStories110MVariant(name: String) -> Bool {
        let normalized = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized == stories110mToken || normalized.contains(stories110mToken)
    }

    /// Convenience overload for `MultiModelConfig`.
    public static func isStories110MVariant(_ config: MultiModelConfig) -> Bool {
        isStories110MVariant(name: config.name)
    }

    /// Stable substring used for Qwen-family artifacts (Qwen2, Qwen2.5, Qwen3, …).
    public static let qwenToken = "qwen"

    /// Returns true when `name` identifies a Qwen-family artifact.
    ///
    /// Matches after trim + lowercase substring, for example:
    /// - `qwen2.5`
    /// - `Qwen2.5-0.5B-Instruct`
    public static func isQwenVariant(name: String) -> Bool {
        let normalized = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return normalized.contains(qwenToken)
    }

    /// Convenience overload for `MultiModelConfig`.
    public static func isQwenVariant(_ config: MultiModelConfig) -> Bool {
        isQwenVariant(name: config.name)
    }

    /// Stable substring used for Qwen2.5-1.5B artifacts.
    public static let qwen15bToken = "1.5b"

    /// Returns true when `name` identifies a Qwen 1.5B-family artifact.
    ///
    /// Matches after trim + lowercase, for example:
    /// - `Qwen2.5-1.5B-Instruct`
    /// - `qwen2.5-1.5b`
    ///
    /// Does not match `Qwen2.5-0.5B-Instruct`.
    public static func isQwen15BVariant(name: String) -> Bool {
        let normalized = name
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return isQwenVariant(name: normalized) && normalized.contains(qwen15bToken)
    }

    /// Convenience overload for `MultiModelConfig`.
    public static func isQwen15BVariant(_ config: MultiModelConfig) -> Bool {
        isQwen15BVariant(name: config.name)
    }
}
