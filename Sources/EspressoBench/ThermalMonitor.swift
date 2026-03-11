import Foundation

enum ThermalMonitor {
    static func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func currentState() -> String {
        thermalStateString(ProcessInfo.processInfo.thermalState)
    }
}
