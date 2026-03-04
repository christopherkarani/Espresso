import Darwin
import Foundation

/// Deterministic sampling matching the ObjC `srand48`/`drand48` sequence.
/// Maps to `train_large.m:317, 374-377`.
public enum Sampler {
    private static let lock = NSLock()

    /// Seed the global drand48 RNG with `srand48(42 + startStep)`.
    public static func seed(startStep: Int) {
        lock.lock()
        srand48(Int(42 + startStep))
        lock.unlock()
    }

    /// Sample a position in `[0, maxPos)`.
    /// ObjC equivalent: `(size_t)(drand48() * max_pos)`.
    public static func samplePosition(maxPos: Int) -> Int {
        precondition(maxPos >= 0)
        lock.lock()
        let r = drand48()
        lock.unlock()
        return Int(r * Double(maxPos))
    }
}

