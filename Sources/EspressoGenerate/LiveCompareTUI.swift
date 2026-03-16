import Darwin
import Foundation

enum LiveLaneStatus: String, Sendable {
    case waiting = "WAITING"
    case compiling = "COMPILING"
    case generating = "GENERATING"
    case completed = "COMPLETED"
    case failed = "FAILED"
}

struct LiveLaneSnapshot: Sendable {
    let title: String
    var status: LiveLaneStatus
    var generatedTokenCount: Int
    var maxTokens: Int
    var lastToken: String
    var text: String
    var compileMs: Double
    var ttftMs: Double
    var tokensPerSecond: Double
    var medianTokenMs: Double
    var p95TokenMs: Double
    var totalMs: Double
    var power: PowerSummary?

    init(title: String, maxTokens: Int) {
        self.title = title
        self.status = .waiting
        self.generatedTokenCount = 0
        self.maxTokens = maxTokens
        self.lastToken = "—"
        self.text = ""
        self.compileMs = 0
        self.ttftMs = 0
        self.tokensPerSecond = 0
        self.medianTokenMs = 0
        self.p95TokenMs = 0
        self.totalMs = 0
        self.power = nil
    }
}

struct LiveCompareSnapshot: Sendable {
    var modelName: String
    var prompt: String
    var maxTokens: Int
    var elapsedMs: Double
    var espresso: LiveLaneSnapshot
    var coreML: LiveLaneSnapshot
    var livePower: PowerSummary?
    var matchCount: Int
    var totalComparedTokens: Int
    var events: [String]
}

final class LiveCompareStateStore: @unchecked Sendable {
    private let lock = NSLock()
    private var snapshot: LiveCompareSnapshot

    init(snapshot: LiveCompareSnapshot) {
        self.snapshot = snapshot
    }

    func mutate(_ body: (inout LiveCompareSnapshot) -> Void) {
        lock.lock()
        body(&snapshot)
        lock.unlock()
    }

    func read() -> LiveCompareSnapshot {
        lock.lock()
        let snapshot = self.snapshot
        lock.unlock()
        return snapshot
    }
}

struct LiveCompareRenderer: Sendable {
    func render(snapshot: LiveCompareSnapshot, size: TerminalSize) -> String {
        let width = max(size.width, 100)
        let laneWidth = max(46, (width - 7) / 2)
        let fullWidth = laneWidth * 2 + 7

        let espressoHeader = colored("ESPRESSO / ANE", .cyan, bold: true)
        let coreHeader = colored("CORE ML", .yellow, bold: true)
        let title = colored("ESPRESSO vs CORE ML LIVE GPT-2", .white, bold: true)
        let promptLine = "prompt: \"\(truncate(snapshot.prompt, to: fullWidth - 24))\""
        let metaLine = "model: \(snapshot.modelName)   tokens: \(snapshot.maxTokens)   elapsed: \(formatDuration(snapshot.elapsedMs))"
        let matchLine = comparisonLine(snapshot)

        var lines: [String] = []
        lines.append(doubleLine(fullWidth))
        lines.append("║ \(padRight(title, to: fullWidth - 4)) ║")
        lines.append("║ \(padRight(promptLine + spaces(4) + metaLine, to: fullWidth - 4)) ║")
        lines.append(doubleLine(fullWidth))
        lines.append(contentsOf: zipColumns(
            left: laneCard(header: espressoHeader, lane: snapshot.espresso, width: laneWidth),
            right: laneCard(header: coreHeader, lane: snapshot.coreML, width: laneWidth),
            laneWidth: laneWidth
        ))
        lines.append(singleLine(fullWidth))
        lines.append(contentsOf: zipColumns(
            left: textCard(title: "ESPRESSO TEXT", text: snapshot.espresso.text, width: laneWidth),
            right: textCard(title: "CORE ML TEXT", text: snapshot.coreML.text, width: laneWidth),
            laneWidth: laneWidth
        ))
        lines.append(singleLine(fullWidth))
        lines.append(contentsOf: powerCard(snapshot: snapshot, width: fullWidth))
        lines.append(singleLine(fullWidth))
        lines.append(contentsOf: eventsCard(snapshot: snapshot, width: fullWidth))
        lines.append(singleLine(fullWidth))
        lines.append("║ \(padRight(matchLine, to: fullWidth - 4)) ║")
        lines.append("║ \(padRight("Ctrl-C to quit", to: fullWidth - 4)) ║")
        lines.append(doubleLine(fullWidth))
        return lines.joined(separator: "\n")
    }

    private func laneCard(header: String, lane: LiveLaneSnapshot, width: Int) -> [String] {
        let barWidth = max(12, width - 28)
        return [
            boxedRow(header, width: width),
            boxedRow("status      \(statusChip(lane.status))", width: width),
            boxedRow("token       \(lane.generatedTokenCount) / \(lane.maxTokens)", width: width),
            boxedRow("last        \(truncate(lane.lastToken, to: width - 13))", width: width),
            boxedRow("", width: width),
            boxedRow(heroMetric(label: "TOKENS / SEC", value: formatDouble(lane.tokensPerSecond)), width: width),
            boxedRow("            \(gradientBar(value: lane.tokensPerSecond, scale: 40, width: barWidth))", width: width),
            boxedRow("TTFT        \(formatDouble(lane.ttftMs)) ms", width: width),
            boxedRow("compile     \(formatDouble(lane.compileMs)) ms", width: width),
            boxedRow("median tok  \(formatDouble(lane.medianTokenMs)) ms", width: width),
            boxedRow("p95 tok     \(formatDouble(lane.p95TokenMs)) ms", width: width),
            boxedRow("runtime     \(formatDouble(lane.totalMs)) ms", width: width),
        ]
    }

    private func textCard(title: String, text: String, width: Int) -> [String] {
        let header = colored(title, .white, bold: true)
        let wrapped = wrap(text.isEmpty ? " " : text, width: width - 4, maxLines: 6)
        var rows = [boxedRow(header, width: width)]
        rows.append(contentsOf: wrapped.map { boxedRow($0, width: width) })
        while rows.count < 7 {
            rows.append(boxedRow("", width: width))
        }
        return rows
    }

    private func powerCard(snapshot: LiveCompareSnapshot, width: Int) -> [String] {
        let live = snapshot.livePower ?? .unavailable
        let espresso = snapshot.espresso.power ?? .unavailable
        let coreML = snapshot.coreML.power ?? .unavailable
        let barWidth = max(24, width - 32)

        return [
            boxedWideRow(colored("POWER", .green, bold: true), width: width),
            boxedWideRow(powerLine(label: "SYSTEM TOTAL", watts: live.packageW, barWidth: barWidth), width: width),
            boxedWideRow(powerLine(label: "ANE", watts: live.aneW, barWidth: barWidth), width: width),
            boxedWideRow(powerLine(label: "CPU", watts: live.cpuW, barWidth: barWidth), width: width),
            boxedWideRow(powerLine(label: "GPU", watts: live.gpuW, barWidth: barWidth), width: width),
            boxedWideRow(
                "Espresso preflight avg: \(formatDouble(espresso.packageW)) W   Core ML preflight avg: \(formatDouble(coreML.packageW)) W",
                width: width
            ),
        ]
    }

    private func eventsCard(snapshot: LiveCompareSnapshot, width: Int) -> [String] {
        let visibleEvents = Array(snapshot.events.suffix(4))
        var rows = [boxedWideRow(colored("EVENT FEED", .magenta, bold: true), width: width)]
        rows.append(contentsOf: visibleEvents.map { boxedWideRow($0, width: width) })
        while rows.count < 5 {
            rows.append(boxedWideRow("", width: width))
        }
        return rows
    }

    private func comparisonLine(_ snapshot: LiveCompareSnapshot) -> String {
        let speedup: String
        if snapshot.coreML.tokensPerSecond > 0 {
            speedup = String(format: "speedup %.2fx", snapshot.espresso.tokensPerSecond / max(snapshot.coreML.tokensPerSecond, 1e-9))
        } else {
            speedup = "speedup —"
        }
        let match = snapshot.totalComparedTokens == 0 ? "match —" : "match \(snapshot.matchCount)/\(snapshot.totalComparedTokens)"
        return "\(speedup)   \(match)"
    }

    private func powerLine(label: String, watts: Double, barWidth: Int) -> String {
        let coloredBar = watts > 0 ? gradientBar(value: watts, scale: 10, width: barWidth) : dim(String(repeating: "░", count: barWidth))
        return "\(label.padding(toLength: 13, withPad: " ", startingAt: 0)) \(coloredBar)  \(formatDouble(watts)) W"
    }

    private func statusChip(_ status: LiveLaneStatus) -> String {
        switch status {
        case .waiting:
            return colored(status.rawValue, .white)
        case .compiling:
            return colored(status.rawValue, .yellow, bold: true)
        case .generating:
            return colored(status.rawValue, .cyan, bold: true)
        case .completed:
            return colored(status.rawValue, .green, bold: true)
        case .failed:
            return colored(status.rawValue, .red, bold: true)
        }
    }

    private func heroMetric(label: String, value: String) -> String {
        "\(colored(label, .white, bold: true)) \(colored(value, .cyan, bold: true))"
    }

    private func gradientBar(value: Double, scale: Double, width: Int) -> String {
        let ratio = min(max(value / max(scale, 1e-9), 0), 1)
        let filled = Int((Double(width) * ratio).rounded(.toNearestOrAwayFromZero))
        var pieces: [String] = []
        for index in 0..<width {
            if index < filled {
                let color: ANSIColor
                let progress = Double(index) / Double(max(width - 1, 1))
                if progress < 0.5 {
                    color = .green
                } else if progress < 0.8 {
                    color = .yellow
                } else {
                    color = .red
                }
                pieces.append(colorize("█", color))
            } else {
                pieces.append(dim("░"))
            }
        }
        return pieces.joined()
    }

    private func wrap(_ text: String, width: Int, maxLines: Int) -> [String] {
        guard width > 8 else { return [truncate(text, to: width)] }
        var remaining = text
        var lines: [String] = []
        while !remaining.isEmpty && lines.count < maxLines {
            if remaining.count <= width {
                lines.append(remaining)
                remaining = ""
                break
            }
            let candidateIndex = remaining.index(remaining.startIndex, offsetBy: width)
            let prefix = String(remaining[..<candidateIndex])
            if let split = prefix.lastIndex(of: " ") {
                let line = String(prefix[..<split])
                lines.append(line)
                remaining = String(remaining[remaining.index(after: split)...]).trimmingCharacters(in: .whitespaces)
            } else {
                lines.append(prefix)
                remaining = String(remaining[candidateIndex...]).trimmingCharacters(in: .whitespaces)
            }
        }
        if !remaining.isEmpty, var last = lines.popLast() {
            last = truncate(last + "…", to: width)
            lines.append(last)
        }
        if lines.isEmpty {
            lines.append("")
        }
        return lines
    }

    private func boxedRow(_ content: String, width: Int) -> String {
        "║ \(padRight(content, to: width - 4)) ║"
    }

    private func boxedWideRow(_ content: String, width: Int) -> String {
        "║ \(padRight(content, to: width - 4)) ║"
    }

    private func zipColumns(left: [String], right: [String], laneWidth: Int) -> [String] {
        let height = max(left.count, right.count)
        return (0..<height).map { index in
            let leftRow = index < left.count ? left[index] : boxedRow("", width: laneWidth)
            let rightRow = index < right.count ? right[index] : boxedRow("", width: laneWidth)
            return "\(leftRow) \(rightRow)"
        }
    }

    private func singleLine(_ width: Int) -> String {
        "╟" + String(repeating: "─", count: width - 2) + "╢"
    }

    private func doubleLine(_ width: Int) -> String {
        "╔" + String(repeating: "═", count: width - 2) + "╗"
    }
}

struct TerminalSize {
    let width: Int
    let height: Int

    static func current() -> TerminalSize {
        var windowSize = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &windowSize) == 0, windowSize.ws_col > 0, windowSize.ws_row > 0 {
            return TerminalSize(width: Int(windowSize.ws_col), height: Int(windowSize.ws_row))
        }
        return TerminalSize(width: 140, height: 40)
    }
}

final class TerminalDisplay: @unchecked Sendable {
    private var active = false

    func start() {
        guard isatty(STDOUT_FILENO) == 1 else { return }
        active = true
        FileHandle.standardOutput.write(Data("\u{001B}[?1049h\u{001B}[?25l".utf8))
    }

    func render(_ content: String) {
        guard active else { return }
        let frame = "\u{001B}[H\u{001B}[2J" + content
        FileHandle.standardOutput.write(Data(frame.utf8))
    }

    func stop() {
        guard active else { return }
        active = false
        FileHandle.standardOutput.write(Data("\u{001B}[?25h\u{001B}[?1049l".utf8))
    }

    deinit {
        stop()
    }
}

private enum ANSIColor: Int {
    case red = 31
    case green = 32
    case yellow = 33
    case blue = 34
    case magenta = 35
    case cyan = 36
    case white = 37
}

private func colored(_ text: String, _ color: ANSIColor, bold: Bool = false) -> String {
    let prefix = bold ? "\u{001B}[1;\(color.rawValue)m" : "\u{001B}[\(color.rawValue)m"
    return prefix + text + "\u{001B}[0m"
}

private func colorize(_ text: String, _ color: ANSIColor) -> String {
    colored(text, color)
}

private func dim(_ text: String) -> String {
    "\u{001B}[2m" + text + "\u{001B}[0m"
}

private func padRight(_ value: String, to width: Int) -> String {
    if visibleWidth(of: value) >= width {
        return truncate(value, to: width)
    }
    return value + spaces(width - visibleWidth(of: value))
}

private func truncate(_ value: String, to width: Int) -> String {
    guard width > 0 else { return "" }
    var visible = 0
    var result = ""
    var iterator = value.makeIterator()
    while let scalar = iterator.next() {
        result.append(scalar)
        if scalar == "\u{001B}" {
            while let next = iterator.next() {
                result.append(next)
                if next == "m" {
                    break
                }
            }
            continue
        }
        visible += 1
        if visible >= width {
            break
        }
    }
    return result
}

private func visibleWidth(of value: String) -> Int {
    var width = 0
    var escape = false
    for scalar in value {
        if escape {
            if scalar == "m" {
                escape = false
            }
            continue
        }
        if scalar == "\u{001B}" {
            escape = true
            continue
        }
        width += 1
    }
    return width
}

private func spaces(_ count: Int) -> String {
    String(repeating: " ", count: max(count, 0))
}

private func formatDouble(_ value: Double) -> String {
    guard value.isFinite else { return "—" }
    return String(format: "%.1f", value)
}

private func formatDuration(_ elapsedMs: Double) -> String {
    let totalSeconds = Int(elapsedMs / 1_000)
    let minutes = totalSeconds / 60
    let seconds = totalSeconds % 60
    return String(format: "%02d:%02d", minutes, seconds)
}
