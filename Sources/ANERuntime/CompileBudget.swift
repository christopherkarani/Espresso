import ANEInterop
import ANETypes

public enum CompileBudget {
    /// Maximum number of ANE compilations before exec() restart is required.
    public static let maxCompiles: Int = ModelConfig.maxCompiles

    /// Current compile count (reads C-level atomic counter).
    public static var currentCount: Int {
        Int(ane_interop_compile_count())
    }

    /// Whether the compile budget is exhausted.
    public static var isExhausted: Bool {
        currentCount >= maxCompiles
    }

    /// Set the compile count (for testing and exec-restart recovery).
    public static func setCount(_ value: Int) throws(ANEError) {
        guard value >= 0 && value <= Int(Int32.max) else {
            throw .invalidArguments("Compile count must be in 0...\(Int32.max), got \(value)")
        }
        ane_interop_set_compile_count(Int32(value))
    }

    /// Remaining compiles before budget exhaustion.
    public static var remaining: Int {
        max(0, maxCompiles - currentCount)
    }
}
