import Foundation

/// Centralized model-family recognition for runtime policy defaults.
///
/// Stories 110M is the retained public demo / benchmark artifact. Several ANE
/// serving defaults (hybrid cached bindings, fused exact head, ANE classifier
/// allowlist) currently prefer this family while broader Llama policy is still
/// opt-in. Keep every name-based check here so special cases do not scatter.
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
}
