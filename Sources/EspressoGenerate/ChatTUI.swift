import Darwin
import Foundation
import ModelSupport

enum ChatLaneStatus: String, Sendable {
    case idle = "IDLE"
    case generating = "GENERATING"
    case cancelled = "CANCELLED"
}

enum ChatPowerFooter: Equatable, Sendable {
    case unavailable(String)
    case pending
    case measured(packageW: Double, cpuW: Double, gpuW: Double, aneW: Double, joulesPerToken: Double?)

    func render() -> String {
        switch self {
        case let .unavailable(message):
            return message
        case .pending:
            return "power: sampling"
        case let .measured(packageW, cpuW, _, aneW, joulesPerToken):
            let watts = String(
                format: "ANE %.2fW  CPU %.2fW  pkg %.2fW",
                aneW,
                cpuW,
                packageW
            )
            if let joulesPerToken, joulesPerToken.isFinite {
                return watts + String(format: "  %.3f J/tok", joulesPerToken)
            }
            return watts
        }
    }
}

func chatPowerFooter(
    capability: PowerCapability,
    summary: PowerSummary?,
    tokensPerSecond: Double
) -> ChatPowerFooter {
    if !capability.available {
        return .unavailable(chatPowerUnavailableMessage(capability))
    }
    guard let summary, summary.sampleCount > 0 else {
        return .unavailable("power: unavailable")
    }
    return .measured(
        packageW: summary.packageW,
        cpuW: summary.cpuW,
        gpuW: summary.gpuW,
        aneW: summary.aneW,
        joulesPerToken: joulesPerToken(packageWatts: summary.packageW, tokensPerSecond: tokensPerSecond)
    )
}

func chatPowerUnavailableMessage(_ capability: PowerCapability) -> String {
    let message = capability.message.trimmingCharacters(in: .whitespacesAndNewlines)
    let lowered = message.lowercased()
    if lowered.contains("sudo") || lowered.contains("passwordless") || lowered.contains("root") {
        return "power: unavailable (sudo)"
    }
    if message.isEmpty {
        return "power: unavailable"
    }
    return message
}

struct ChatStatusFooter: Equatable, Sendable {
    var tokensPerSecond: Double
    var ttftMs: Double
    var decodePath: String
    var contextUsed: Int
    var contextMax: Int
    var power: ChatPowerFooter = .unavailable("power: unavailable")

    func render() -> String {
        let tok = tokensPerSecond.isFinite ? String(format: "%.1f", tokensPerSecond) : "—"
        let ttft = ttftMs.isFinite ? String(format: "%.0f", ttftMs) : "—"
        let metrics = "tok/s \(tok)  TTFT \(ttft)ms  path=\(decodePath)  ctx \(contextUsed)/\(contextMax)"
        return metrics + "\n" + power.render()
    }

    func renderLines() -> [String] {
        render().split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    }
}

struct ChatSnapshot: Sendable {
    var modelName: String
    var turns: [ChatTurn]
    var streamingAssistant: String
    var status: ChatLaneStatus
    var footer: ChatStatusFooter
}

struct ChatTUIRenderer: Sendable {
    func render(snapshot: ChatSnapshot, size: TerminalSize) -> String {
        let width = max(size.width, 60)
        var lines: [String] = []
        lines.append(doubleLine(width))
        lines.append(boxed("ESPRESSO CHAT  \(snapshot.modelName)", width: width))
        lines.append(singleLine(width))

        let bodyHeight = max(size.height - 9, 6)
        var body: [String] = []
        for turn in snapshot.turns {
            let label = turn.role == .user ? "you" : "qwen"
            body.append(contentsOf: wrapTurn(label: label, text: turn.content, width: width - 4))
            body.append("")
        }
        if snapshot.status == .generating || !snapshot.streamingAssistant.isEmpty {
            body.append(contentsOf: wrapTurn(label: "qwen", text: snapshot.streamingAssistant, width: width - 4))
            body.append("")
        }
        if body.isEmpty {
            body.append("type a message, or /reset /retry /exit")
        }
        let visible = Array(body.suffix(bodyHeight))
        for row in visible {
            lines.append(boxed(row, width: width))
        }
        while lines.count < bodyHeight + 3 {
            lines.append(boxed("", width: width))
        }

        lines.append(singleLine(width))
        for line in snapshot.footer.renderLines() {
            lines.append(boxed(line, width: width))
        }
        lines.append(boxed("status \(snapshot.status.rawValue)   Ctrl-C cancels the current completion", width: width))
        lines.append(doubleLine(width))
        return lines.joined(separator: "\n")
    }

    private func wrapTurn(label: String, text: String, width: Int) -> [String] {
        let prefix = "\(label) · "
        let wrapped = wrap(text.isEmpty ? "…" : text, width: max(width - prefix.count, 8), maxLines: 8)
        return wrapped.enumerated().map { index, line in
            index == 0 ? prefix + line : String(repeating: " ", count: prefix.count) + line
        }
    }

    private func wrap(_ text: String, width: Int, maxLines: Int) -> [String] {
        guard width > 4 else { return [String(text.prefix(width))] }
        var remaining = text
        var lines: [String] = []
        while !remaining.isEmpty && lines.count < maxLines {
            if remaining.count <= width {
                lines.append(remaining)
                remaining = ""
                break
            }
            let end = remaining.index(remaining.startIndex, offsetBy: width)
            let prefix = String(remaining[..<end])
            if let split = prefix.lastIndex(of: " ") {
                lines.append(String(prefix[..<split]))
                remaining = String(remaining[remaining.index(after: split)...])
                    .trimmingCharacters(in: .whitespaces)
            } else {
                lines.append(prefix)
                remaining = String(remaining[end...])
            }
        }
        if !remaining.isEmpty, var last = lines.popLast() {
            last = String(last.prefix(max(width - 1, 1))) + "…"
            lines.append(last)
        }
        return lines.isEmpty ? [""] : lines
    }

    private func boxed(_ content: String, width: Int) -> String {
        let inner = width - 4
        let clipped = content.count <= inner ? content : String(content.prefix(inner))
        let pad = max(inner - clipped.count, 0)
        return "║ \(clipped)\(String(repeating: " ", count: pad)) ║"
    }

    private func singleLine(_ width: Int) -> String {
        "╟" + String(repeating: "─", count: width - 2) + "╢"
    }

    private func doubleLine(_ width: Int) -> String {
        "╔" + String(repeating: "═", count: width - 2) + "╗"
    }
}

final class ChatCancelState: @unchecked Sendable {
    private let lock = NSLock()
    private var generating = false
    private var cancelled = false

    private var userInterrupt = false

    func beginGeneration() {
        lock.lock()
        generating = true
        cancelled = false
        userInterrupt = false
        lock.unlock()
    }

    func endGeneration() {
        lock.lock()
        generating = false
        lock.unlock()
    }

    @discardableResult
    func requestCancel(userInterrupt: Bool = false) -> Bool {
        lock.lock()
        let wasGenerating = generating
        if wasGenerating {
            cancelled = true
            if userInterrupt {
                self.userInterrupt = true
            }
        }
        lock.unlock()
        return wasGenerating
    }

    var wasUserInterrupt: Bool {
        lock.lock()
        let value = userInterrupt
        lock.unlock()
        return value
    }

    var isCancelled: Bool {
        lock.lock()
        let value = cancelled
        lock.unlock()
        return value
    }

    var isGenerating: Bool {
        lock.lock()
        let value = generating
        lock.unlock()
        return value
    }
}
