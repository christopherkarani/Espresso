import ANETypes
import Foundation
import ModelSupport

enum ChatCommand: Equatable, Sendable {
    case message(String)
    case reset
    case retry
    case exit
    case empty

    static func parse(_ line: String) -> ChatCommand {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        switch trimmed {
        case "/reset":
            return .reset
        case "/retry":
            return .retry
        case "/exit", "/quit":
            return .exit
        case "":
            return .empty
        default:
            return .message(trimmed)
        }
    }
}

enum ChatSessionAction: Equatable, Sendable {
    case generate(String)
    case exit
    case noop(String)
}

struct ChatSampling: Equatable, Sendable {
    var temperature: Float
    var topP: Float
    var isGreedy: Bool
}

struct ChatSession: Equatable, Sendable {
    var system: String
    var messages: [ChatTurn]

    init(system: String = QwenInstructPrompt.systemBanner) {
        self.system = system.isEmpty ? QwenInstructPrompt.systemBanner : system
        self.messages = []
    }

    mutating func apply(_ command: ChatCommand) -> ChatSessionAction {
        switch command {
        case .empty:
            return .noop("")
        case .exit:
            return .exit
        case .reset:
            messages.removeAll()
            return .noop("history cleared")
        case .retry:
            guard let lastUserIndex = messages.lastIndex(where: { $0.role == .user }) else {
                return .noop("nothing to retry")
            }
            messages = Array(messages.prefix(lastUserIndex + 1))
            return .generate(renderPrompt())
        case let .message(text):
            messages.append(ChatTurn(role: .user, content: text))
            return .generate(renderPrompt())
        }
    }

    mutating func appendAssistant(_ text: String) {
        let trimmed = Self.sanitizeAssistantText(text)
        guard !trimmed.isEmpty else { return }
        if messages.last?.role == .assistant {
            messages[messages.count - 1] = ChatTurn(role: .assistant, content: trimmed)
        } else {
            messages.append(ChatTurn(role: .assistant, content: trimmed))
        }
    }

    func renderPrompt() -> String {
        QwenInstructPrompt.render(messages: messages, system: system, addGenerationPrompt: true)
    }

    static func sanitizeAssistantText(_ text: String) -> String {
        var result = text
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"] {
            if let range = result.range(of: marker) {
                result = String(result[..<range.lowerBound])
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

func resolvedSampling(command: CommandName, options: Options) -> ChatSampling {
    if options.greedy {
        return ChatSampling(temperature: 0, topP: 1, isGreedy: true)
    }
    if command == .chat {
        return ChatSampling(
            temperature: options.temperatureWasSet ? options.temperature : 0.7,
            topP: options.topPWasSet ? options.topP : 0.9,
            isGreedy: false
        )
    }
    return ChatSampling(
        temperature: options.temperature,
        topP: options.topPWasSet ? options.topP : 1,
        isGreedy: options.temperature <= 0
    )
}

func chatForcesHybridFallbackDisable(_ options: Options) -> Bool {
    options.command == .chat
}

func assertChatDecodePathIsHybrid(_ path: String) throws {
    let normalized = path.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard normalized == "hybrid" else {
        throw CLIError.runtime(
            "chat requires path=hybrid (ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1); got \(path)"
        )
    }
}

func assistantTextFromGeneratedTokens(
    _ tokens: [TokenID],
    decode: ([Int]) -> String,
    eosToken: TokenID?
) -> String {
    let filtered = tokens.filter { token in
        if let eosToken, token == eosToken {
            return false
        }
        return true
    }
    return decode(filtered.map(Int.init)).trimmingCharacters(in: .whitespacesAndNewlines)
}
