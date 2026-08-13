/// Official Qwen2.5-Instruct chat wrap (no Hugging Face / Jinja).
///
/// Matches the tokenizer `apply_chat_template` contract used by the greedy
/// fixture: system banner, one user turn, then the assistant generation prompt.
public enum QwenInstructPrompt: Sendable {
    public static let systemBanner =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    /// Wraps a single user turn in the Qwen2.5-Instruct chat template.
    public static func wrapUserTurn(_ prompt: String) -> String {
        "<|im_start|>system\n\(systemBanner)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// True when generate should apply `wrapUserTurn` unless `--raw-prompt` is set.
    public static func shouldWrap(config: MultiModelConfig) -> Bool {
        ModelFamily.isQwenVariant(config)
    }
}
