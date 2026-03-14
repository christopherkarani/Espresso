/// 4D tensor shape for ANE: [batch, channels, height, spatial].
///
/// ANE always operates on 4D tensors. For typical LLM workloads:
/// - batch = 1 (always)
/// - channels = model dimension, head count, vocab size, etc.
/// - height = 1 for decode, head_dim for multi-head attention reshapes
/// - spatial = sequence length or lane width
public struct ANEShape: Sendable, Equatable {
    public var batch: Int
    public var channels: Int
    public var height: Int
    public var spatial: Int

    public init(batch: Int = 1, channels: Int, height: Int = 1, spatial: Int) {
        self.batch = batch
        self.channels = channels
        self.height = height
        self.spatial = spatial
    }

    /// Total number of elements in the tensor.
    public var elementCount: Int {
        batch * channels * height * spatial
    }

    /// Total byte size for a given data type.
    public func byteSize(for dtype: ANEDType) -> Int {
        elementCount * dtype.byteWidth
    }

    /// Shape as a 4-element array [batch, channels, height, spatial].
    public var dimensions: [Int] {
        [batch, channels, height, spatial]
    }

    /// Whether this shape meets the ANE minimum IOSurface size (49,152 bytes / ~48KB).
    /// Tensors smaller than this fail ANE eval with status 0x1d.
    public func meetsMinimumIOSurfaceSize(for dtype: ANEDType) -> Bool {
        byteSize(for: dtype) >= 49_152
    }

    /// Whether the byte size exceeds ANE's 32MB on-chip SRAM budget.
    /// Exceeding this causes ~30% throughput drop due to DRAM spill.
    public func exceedsSRAMBudget(for dtype: ANEDType) -> Bool {
        byteSize(for: dtype) > 32 * 1024 * 1024
    }
}
