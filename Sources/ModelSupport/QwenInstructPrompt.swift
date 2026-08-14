/// Official Qwen2.5-Instruct chat wrap (no Hugging Face / Jinja).
///
/// Matches the tokenizer `apply_chat_template` contract used by the greedy
/// fixture: system banner, one user turn, then the assistant generation prompt.
public enum ChatRole: String, Sendable, Equatable {
    case system
    case user
    case assistant
}

public struct ChatTurn: Sendable, Equatable {
    public var role: ChatRole
    public var content: String

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

public enum QwenInstructPrompt: Sendable {
    public static let systemBanner =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    /// Wraps a single user turn in the Qwen2.5-Instruct chat template.
    public static func wrapUserTurn(_ prompt: String) -> String {
        render(messages: [ChatTurn(role: .user, content: prompt)])
    }

    /// Renders a multi-turn Qwen2.5-Instruct conversation.
    ///
    /// History is re-emitted in full so a later turn cannot drop an earlier one.
    /// When `addGenerationPrompt` is true the string ends with the assistant
    /// generation prefix, matching Hugging Face `apply_chat_template`.
    public static func render(
        messages: [ChatTurn],
        system: String? = nil,
        addGenerationPrompt: Bool = true
    ) -> String {
        let systemText = (system?.isEmpty == false) ? system! : systemBanner
        var parts: [String] = ["<|im_start|>system\n\(systemText)<|im_end|>\n"]
        for message in messages where message.role != .system {
            parts.append("<|im_start|>\(message.role.rawValue)\n\(message.content)<|im_end|>\n")
        }
        if addGenerationPrompt {
            parts.append("<|im_start|>assistant\n")
        }
        return parts.joined()
    }

    /// True when generate should apply `wrapUserTurn` unless `--raw-prompt` is set.
    public static func shouldWrap(config: MultiModelConfig) -> Bool {
        ModelFamily.isQwenVariant(config)
    }
}
