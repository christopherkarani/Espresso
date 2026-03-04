import Dispatch
import ANETypes

/// Serial queue + dispatch group used to overlap CPU dW accumulation with ANE eval.
/// `@unchecked Sendable` is safe because all mutation is serialized by the internal queue.
public final class GradientAccumulator: @unchecked Sendable {
    private let queue: DispatchQueue
    private let group: DispatchGroup

    public init() {
        self.queue = DispatchQueue(label: "dw_cblas", qos: .userInitiated)
        self.group = DispatchGroup()
    }

    public func enqueue(_ block: @escaping @Sendable () -> Void) {
        group.enter()
        queue.async {
            block()
            self.group.leave()
        }
    }

    /// Barrier for all previously enqueued work.
    /// Maps to `dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER)`.
    public func barrier() {
        _ = group.wait(timeout: .distantFuture)
    }

    /// Wait for all enqueued work to finish.
    public func waitAll() {
        barrier()
    }
}

/// Owns an exclusive heap copy of a float buffer.
/// ~Copyable prevents accidental aliasing/copying across async blocks.
public struct SendableBuffer: ~Copyable, @unchecked Sendable {
    public let pointer: UnsafeMutablePointer<Float>
    public let count: Int

    public init(copying source: borrowing TensorBuffer) {
        self.count = source.count
        self.pointer = .allocate(capacity: count)
        source.withUnsafePointer { src in
            self.pointer.initialize(from: src, count: count)
        }
    }

    public init(copying source: UnsafePointer<Float>, count: Int) {
        precondition(count >= 0)
        self.count = count
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(from: source, count: count)
    }

    deinit {
        pointer.deinitialize(count: count)
        pointer.deallocate()
    }
}

/// Sendable mutable pointer wrapper for gradient accumulator captures.
/// Safety relies on the accumulator's serial queue providing exclusive access within a batch.
public struct SendablePointer: @unchecked Sendable {
    public let pointer: UnsafeMutablePointer<Float>
    public init(_ pointer: UnsafeMutablePointer<Float>) { self.pointer = pointer }
}

/// Sendable const pointer wrapper for read-only captures (e.g. dembed block).
public struct SendableConstPointer: @unchecked Sendable {
    public let pointer: UnsafePointer<Float>
    public init(_ pointer: UnsafePointer<Float>) { self.pointer = pointer }
}
