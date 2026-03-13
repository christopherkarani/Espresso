import IOSurface
import ANEInterop

public enum SurfaceIOError: Error, Equatable {
    case argumentOutOfRange
    case interopCallFailed
}

public enum SurfaceIO {
    public struct FP16ArgmaxResult: Equatable {
        public let index: Int
        public let value: Float

        public init(index: Int, value: Float) {
            self.index = index
            self.value = value
        }
    }

    public struct FP16ReadRegion {
        public let destination: UnsafeMutablePointer<Float>
        public let channelOffset: Int
        public let channels: Int

        public init(destination: UnsafeMutablePointer<Float>, channelOffset: Int, channels: Int) {
            self.destination = destination
            self.channelOffset = channelOffset
            self.channels = channels
        }
    }

    public struct FP16WriteRegion {
        public let source: UnsafePointer<Float>
        public let channelOffset: Int
        public let channels: Int

        public init(source: UnsafePointer<Float>, channelOffset: Int, channels: Int) {
            self.source = source
            self.channelOffset = channelOffset
            self.channels = channels
        }
    }

    public struct FP16CopyRegion {
        public let dstChannelOffset: Int
        public let srcChannelOffset: Int
        public let channels: Int

        public init(dstChannelOffset: Int, srcChannelOffset: Int, channels: Int) {
            self.dstChannelOffset = dstChannelOffset
            self.srcChannelOffset = srcChannelOffset
            self.channels = channels
        }
    }

    public struct FP16SourceCopyRegion {
        public let source: IOSurfaceRef
        public let dstChannelOffset: Int
        public let srcChannelOffset: Int
        public let channels: Int

        public init(source: IOSurfaceRef, dstChannelOffset: Int, srcChannelOffset: Int, channels: Int) {
            self.source = source
            self.dstChannelOffset = dstChannelOffset
            self.srcChannelOffset = srcChannelOffset
            self.channels = channels
        }
    }

    @inline(__always)
    private static func checkedElementCount(channels: Int, spatial: Int) -> Int {
        precondition(channels >= 0 && spatial >= 0)
        let count = channels.multipliedReportingOverflow(by: spatial)
        precondition(!count.overflow)
        return count.partialValue
    }

    @inline(__always)
    private static func checkedNonNegativeInt32(_ value: Int) throws(SurfaceIOError) -> Int32 {
        guard value >= 0, value <= Int(Int32.max) else {
            throw .argumentOutOfRange
        }
        return Int32(value)
    }

    public static func writeFP16(to surface: IOSurfaceRef, data: UnsafeBufferPointer<Float>, channels: Int, spatial: Int) {
        let count = checkedElementCount(channels: channels, spatial: spatial)
        precondition(data.count == count)
        if count == 0 { return }
        guard let src = data.baseAddress else {
            preconditionFailure("Input base address is nil for non-empty buffer")
        }

        let channels32 = Int32(clamping: channels)
        let spatial32 = Int32(clamping: spatial)
        precondition(Int(channels32) == channels && Int(spatial32) == spatial)
        let ok = ane_interop_io_write_fp16(surface, src, channels32, spatial32)
        precondition(ok)
    }

    public static func readFP16(from surface: IOSurfaceRef,
                               into dst: UnsafeMutableBufferPointer<Float>,
                               channelOffset: Int,
                               channels: Int,
                               spatial: Int) {
        let count = checkedElementCount(channels: channels, spatial: spatial)
        precondition(channelOffset >= 0)
        precondition(dst.count == count)
        if count == 0 { return }
        guard let out = dst.baseAddress else {
            preconditionFailure("Destination base address is nil for non-empty buffer")
        }

        let channelOffset32 = Int32(clamping: channelOffset)
        let channels32 = Int32(clamping: channels)
        let spatial32 = Int32(clamping: spatial)
        precondition(Int(channelOffset32) == channelOffset)
        precondition(Int(channels32) == channels && Int(spatial32) == spatial)
        let ok = ane_interop_io_read_fp16(surface, channelOffset32, out, channels32, spatial32)
        precondition(ok)
    }

    /// Read multiple FP16 channel slices from one surface under a single lock.
    ///
    /// Each region destination must point to writable storage of at least `channels * spatial` floats.
    public static func readFP16Batched(from surface: IOSurfaceRef, spatial: Int, regions: [FP16ReadRegion]) {
        precondition(spatial >= 0)
        if regions.isEmpty || spatial == 0 { return }
        let spatial32 = Int32(clamping: spatial)
        precondition(Int(spatial32) == spatial)

        var destinations = [UnsafeMutablePointer<Float>?]()
        var channelOffsets = [Int32]()
        var channelCounts = [Int32]()
        destinations.reserveCapacity(regions.count)
        channelOffsets.reserveCapacity(regions.count)
        channelCounts.reserveCapacity(regions.count)

        for region in regions {
            precondition(region.channelOffset >= 0)
            precondition(region.channels >= 0)
            let offset32 = Int32(clamping: region.channelOffset)
            let channels32 = Int32(clamping: region.channels)
            precondition(Int(offset32) == region.channelOffset)
            precondition(Int(channels32) == region.channels)
            destinations.append(region.destination)
            channelOffsets.append(offset32)
            channelCounts.append(channels32)
        }

        let ok = destinations.withUnsafeMutableBufferPointer { dstBuf in
            channelOffsets.withUnsafeBufferPointer { offBuf in
                channelCounts.withUnsafeBufferPointer { chBuf in
                    let count32 = Int32(clamping: regions.count)
                    precondition(Int(count32) == regions.count)
                    return ane_interop_io_read_fp16_batched(
                        surface,
                        spatial32,
                        dstBuf.baseAddress,
                        offBuf.baseAddress,
                        chBuf.baseAddress,
                        count32
                    )
                }
            }
        }
        precondition(ok)
    }

    public static func writeFP16At(to surface: IOSurfaceRef,
                                   channelOffset: Int,
                                   data: UnsafeBufferPointer<Float>,
                                   channels: Int,
                                   spatial: Int) throws(SurfaceIOError) {
        let channelOffset32 = try checkedNonNegativeInt32(channelOffset)
        let channels32 = try checkedNonNegativeInt32(channels)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let count = checkedElementCount(channels: channels, spatial: spatial)
        precondition(data.count == count)
        if count == 0 { return }

        guard let src = data.baseAddress else {
            preconditionFailure("Input base address is nil for non-empty buffer")
        }

        let ok = ane_interop_io_write_fp16_at(surface, channelOffset32, src, channels32, spatial32)
        guard ok else { throw .interopCallFailed }
    }

    public static func writeFP16AtBatched(to surface: IOSurfaceRef,
                                          spatial: Int,
                                          regions: [FP16WriteRegion]) throws(SurfaceIOError) {
        let spatial32 = try checkedNonNegativeInt32(spatial)
        if regions.isEmpty || spatial == 0 { return }

        var sources = [UnsafePointer<Float>?]()
        var offsets = [Int32]()
        var channelCounts = [Int32]()
        sources.reserveCapacity(regions.count)
        offsets.reserveCapacity(regions.count)
        channelCounts.reserveCapacity(regions.count)

        for region in regions {
            let offset32 = try checkedNonNegativeInt32(region.channelOffset)
            let channels32 = try checkedNonNegativeInt32(region.channels)
            _ = checkedElementCount(channels: region.channels, spatial: spatial)
            sources.append(region.source)
            offsets.append(offset32)
            channelCounts.append(channels32)
        }

        let ok = sources.withUnsafeMutableBufferPointer { srcBuf in
            offsets.withUnsafeMutableBufferPointer { offBuf in
                channelCounts.withUnsafeMutableBufferPointer { chBuf in
                    let count32 = Int32(clamping: regions.count)
                    precondition(Int(count32) == regions.count)
                    return ane_interop_io_write_fp16_at_batched(
                        surface,
                        offBuf.baseAddress,
                        srcBuf.baseAddress,
                        chBuf.baseAddress,
                        count32,
                        spatial32
                    )
                }
            }
        }
        guard ok else { throw .interopCallFailed }
    }

    public static func copyFP16(dst: IOSurfaceRef,
                               dstChannelOffset: Int,
                               src: IOSurfaceRef,
                               srcChannelOffset: Int,
                               channels: Int,
                               spatial: Int) throws(SurfaceIOError) {
        let dstOffset32 = try checkedNonNegativeInt32(dstChannelOffset)
        let srcOffset32 = try checkedNonNegativeInt32(srcChannelOffset)
        let channels32 = try checkedNonNegativeInt32(channels)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        _ = checkedElementCount(channels: channels, spatial: spatial)
        if channels == 0 || spatial == 0 { return }

        let ok = ane_interop_io_copy(dst, dstOffset32, src, srcOffset32, channels32, spatial32)
        guard ok else { throw .interopCallFailed }
    }

    /// Copy a single spatial index (token) across `channels` from `src` into `dst`.
    ///
    /// Surfaces are interpreted as channel-first `[channels, spatial]` FP16 tensors.
    public static func copyFP16SpatialSlice(
        dst: IOSurfaceRef,
        dstChannelOffset: Int,
        dstSpatialIndex: Int,
        dstSpatial: Int,
        src: IOSurfaceRef,
        srcChannelOffset: Int,
        srcSpatialIndex: Int,
        srcSpatial: Int,
        channels: Int
    ) throws(SurfaceIOError) {
        let dstOff32 = try checkedNonNegativeInt32(dstChannelOffset)
        let srcOff32 = try checkedNonNegativeInt32(srcChannelOffset)
        let dstSpatialIndex32 = try checkedNonNegativeInt32(dstSpatialIndex)
        let srcSpatialIndex32 = try checkedNonNegativeInt32(srcSpatialIndex)
        let dstSpatial32 = try checkedNonNegativeInt32(dstSpatial)
        let srcSpatial32 = try checkedNonNegativeInt32(srcSpatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        if channels == 0 { return }

        let ok = ane_interop_io_copy_fp16_spatial_slice(
            dst,
            dstOff32,
            dstSpatialIndex32,
            dstSpatial32,
            src,
            srcOff32,
            srcSpatialIndex32,
            srcSpatial32,
            channels32
        )
        guard ok else { throw .interopCallFailed }
    }

    public static func writeFP16SpatialSlice(
        to surface: IOSurfaceRef,
        channelOffset: Int,
        spatialIndex: Int,
        spatial: Int,
        data: UnsafeBufferPointer<Float>,
        channels: Int
    ) throws(SurfaceIOError) {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatialIndex32 = try checkedNonNegativeInt32(spatialIndex)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        precondition(data.count == channels)
        if channels == 0 { return }
        guard let src = data.baseAddress else {
            preconditionFailure("Input base address is nil for non-empty buffer")
        }

        let ok = ane_interop_io_write_fp16_spatial_slice(
            surface,
            chOff32,
            spatialIndex32,
            spatial32,
            src,
            channels32
        )
        guard ok else { throw .interopCallFailed }
    }

    public static func readFP16SpatialSlice(
        from surface: IOSurfaceRef,
        channelOffset: Int,
        spatialIndex: Int,
        spatial: Int,
        into dst: UnsafeMutableBufferPointer<Float>,
        channels: Int
    ) throws(SurfaceIOError) {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatialIndex32 = try checkedNonNegativeInt32(spatialIndex)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        precondition(dst.count == channels)
        if channels == 0 { return }
        guard let out = dst.baseAddress else {
            preconditionFailure("Destination base address is nil for non-empty buffer")
        }

        let ok = ane_interop_io_read_fp16_spatial_slice(
            surface,
            chOff32,
            spatialIndex32,
            spatial32,
            out,
            channels32
        )
        guard ok else { throw .interopCallFailed }
    }

    public static func argmaxFP16SpatialSlice(
        from surface: IOSurfaceRef,
        channelOffset: Int,
        spatialIndex: Int,
        spatial: Int,
        channels: Int
    ) throws(SurfaceIOError) -> FP16ArgmaxResult {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatialIndex32 = try checkedNonNegativeInt32(spatialIndex)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        guard spatial > 0, channels > 0, spatialIndex < spatial else {
            throw .argumentOutOfRange
        }

        var index32: Int32 = 0
        var value: Float = 0
        let ok = ane_interop_io_argmax_fp16_spatial_slice(
            surface,
            chOff32,
            spatialIndex32,
            spatial32,
            channels32,
            &index32,
            &value
        )
        guard ok else { throw .interopCallFailed }
        return FP16ArgmaxResult(index: Int(index32), value: value)
    }

    /// Write embeddings for multiple streams to their spatial lanes under one lock.
    /// Fuses embedding lookup + FP32→FP16 conversion + strided surface write.
    public static func writeEmbeddingBatchFP16(
        to surface: IOSurfaceRef,
        channelOffset: Int,
        spatial: Int,
        embeddingTable: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        tokenIDs: UnsafePointer<UInt16>,
        streamCount: Int
    ) throws(SurfaceIOError) {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let vocab32 = try checkedNonNegativeInt32(vocabSize)
        let dim32 = try checkedNonNegativeInt32(dim)
        let count32 = try checkedNonNegativeInt32(streamCount)
        guard spatial > 0, vocabSize > 0, dim > 0, streamCount > 0, streamCount <= spatial else {
            throw .argumentOutOfRange
        }
        let ok = ane_interop_io_write_embedding_batch_fp16(
            surface, chOff32, spatial32, embeddingTable, vocab32, dim32, tokenIDs, count32
        )
        guard ok else { throw .interopCallFailed }
    }

    /// Argmax over multiple spatial lanes under one lock.
    public static func argmaxBatchFP16Spatial(
        from surface: IOSurfaceRef,
        channelOffset: Int,
        spatial: Int,
        channels: Int,
        streamCount: Int
    ) throws(SurfaceIOError) -> [FP16ArgmaxResult] {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        let count32 = try checkedNonNegativeInt32(streamCount)
        guard spatial > 0, channels > 0, streamCount > 0, streamCount <= spatial else {
            throw .argumentOutOfRange
        }

        var indices = [Int32](repeating: 0, count: streamCount)
        var values = [Float](repeating: 0, count: streamCount)
        let ok = indices.withUnsafeMutableBufferPointer { idxBuf in
            values.withUnsafeMutableBufferPointer { valBuf in
                ane_interop_io_argmax_batch_fp16_spatial(
                    surface, chOff32, spatial32, channels32, count32,
                    idxBuf.baseAddress!, valBuf.baseAddress!
                )
            }
        }
        guard ok else { throw .interopCallFailed }

        return (0..<streamCount).map { i in
            FP16ArgmaxResult(index: Int(indices[i]), value: values[i])
        }
    }

    public static func argmaxBatchFP16SpatialParallel(
        from surface: IOSurfaceRef,
        channelOffset: Int,
        spatial: Int,
        channels: Int,
        streamCount: Int,
        nBlocks: Int = 4
    ) throws(SurfaceIOError) -> [FP16ArgmaxResult] {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        let count32 = try checkedNonNegativeInt32(streamCount)
        let blocks32 = try checkedNonNegativeInt32(nBlocks)
        guard spatial > 0, channels > 0, streamCount > 0, streamCount <= spatial else {
            throw .argumentOutOfRange
        }
        guard channels <= Int(UInt16.max) + 1 else {
            throw .argumentOutOfRange
        }

        var indices = [Int32](repeating: 0, count: streamCount)
        var values = [Float](repeating: 0, count: streamCount)
        let ok = indices.withUnsafeMutableBufferPointer { idxBuf in
            values.withUnsafeMutableBufferPointer { valBuf in
                ane_interop_io_argmax_batch_fp16_spatial_parallel(
                    surface, chOff32, spatial32, channels32, count32,
                    idxBuf.baseAddress!, valBuf.baseAddress!, blocks32
                )
            }
        }
        guard ok else { throw .interopCallFailed }

        return (0..<streamCount).map { i in
            FP16ArgmaxResult(index: Int(indices[i]), value: values[i])
        }
    }

    /// Lockless argmax — assumes surface is already coherent (e.g., after sync ANE eval).
    public static func argmaxBatchFP16SpatialNolock(
        from surface: IOSurfaceRef,
        channelOffset: Int,
        spatial: Int,
        channels: Int,
        streamCount: Int,
        nBlocks: Int = 8
    ) throws(SurfaceIOError) -> [FP16ArgmaxResult] {
        let chOff32 = try checkedNonNegativeInt32(channelOffset)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let channels32 = try checkedNonNegativeInt32(channels)
        let count32 = try checkedNonNegativeInt32(streamCount)
        let blocks32 = try checkedNonNegativeInt32(nBlocks)
        guard spatial > 0, channels > 0, streamCount > 0, streamCount <= spatial else {
            throw .argumentOutOfRange
        }

        var indices = [Int32](repeating: 0, count: streamCount)
        var values = [Float](repeating: 0, count: streamCount)
        let ok = indices.withUnsafeMutableBufferPointer { idxBuf in
            values.withUnsafeMutableBufferPointer { valBuf in
                ane_interop_io_argmax_batch_fp16_spatial_nolock(
                    surface, chOff32, spatial32, channels32, count32,
                    idxBuf.baseAddress!, valBuf.baseAddress!, blocks32
                )
            }
        }
        guard ok else { throw .interopCallFailed }

        return (0..<streamCount).map { i in
            FP16ArgmaxResult(index: Int(indices[i]), value: values[i])
        }
    }

    /// Fused expansion + argmax: reads a small projected surface, computes
    /// matmul + argmax on CPU without materializing full vocab logits.
    /// `expansionWeightsFP16` must point to contiguous fp16 [vocabSize, bottleneck/groups].
    public static func fusedExpansionArgmax(
        projSurface: IOSurfaceRef,
        projChannelOffset: Int = 0,
        spatial: Int,
        bottleneck: Int,
        groups: Int,
        expansionWeightsFP16: UnsafeRawPointer,
        vocabSize: Int,
        streamCount: Int,
        nBlocks: Int = 8
    ) throws(SurfaceIOError) -> [FP16ArgmaxResult] {
        let projChOff32 = try checkedNonNegativeInt32(projChannelOffset)
        let spatial32 = try checkedNonNegativeInt32(spatial)
        let bottleneck32 = try checkedNonNegativeInt32(bottleneck)
        let groups32 = try checkedNonNegativeInt32(groups)
        let vocab32 = try checkedNonNegativeInt32(vocabSize)
        let count32 = try checkedNonNegativeInt32(streamCount)
        let blocks32 = try checkedNonNegativeInt32(nBlocks)
        guard spatial > 0, bottleneck > 0, groups > 0, vocabSize > 0, streamCount > 0 else {
            throw .argumentOutOfRange
        }
        guard streamCount <= spatial else {
            throw .argumentOutOfRange
        }
        guard bottleneck.isMultiple(of: groups), vocabSize.isMultiple(of: groups) else {
            throw .argumentOutOfRange
        }
        guard vocabSize <= Int(UInt16.max) + 1 else {
            throw .argumentOutOfRange
        }
        var indices = [Int32](repeating: 0, count: streamCount)
        var values = [Float](repeating: 0, count: streamCount)
        let ok = indices.withUnsafeMutableBufferPointer { idxBuf in
            values.withUnsafeMutableBufferPointer { valBuf in
                ane_interop_fused_expansion_argmax_fp16(
                    projSurface,
                    projChOff32,
                    spatial32,
                    bottleneck32,
                    groups32,
                    expansionWeightsFP16,
                    vocab32,
                    count32,
                    idxBuf.baseAddress!,
                    valBuf.baseAddress!,
                    blocks32
                )
            }
        }
        guard ok else { throw .interopCallFailed }
        return (0..<streamCount).map { i in
            FP16ArgmaxResult(index: Int(indices[i]), value: values[i])
        }
    }

    public static func copyFP16Batched(dst: IOSurfaceRef,
                                       src: IOSurfaceRef,
                                       spatial: Int,
                                       regions: [FP16CopyRegion]) throws(SurfaceIOError) {
        let spatial32 = try checkedNonNegativeInt32(spatial)
        if regions.isEmpty || spatial == 0 { return }

        var dstOffsets = [Int32]()
        var srcOffsets = [Int32]()
        var channelCounts = [Int32]()
        dstOffsets.reserveCapacity(regions.count)
        srcOffsets.reserveCapacity(regions.count)
        channelCounts.reserveCapacity(regions.count)

        for region in regions {
            let dstOffset32 = try checkedNonNegativeInt32(region.dstChannelOffset)
            let srcOffset32 = try checkedNonNegativeInt32(region.srcChannelOffset)
            let channels32 = try checkedNonNegativeInt32(region.channels)
            _ = checkedElementCount(channels: region.channels, spatial: spatial)
            dstOffsets.append(dstOffset32)
            srcOffsets.append(srcOffset32)
            channelCounts.append(channels32)
        }

        let ok = dstOffsets.withUnsafeMutableBufferPointer { dstBuf in
            srcOffsets.withUnsafeMutableBufferPointer { srcBuf in
                channelCounts.withUnsafeMutableBufferPointer { chBuf in
                    let count32 = Int32(clamping: regions.count)
                    precondition(Int(count32) == regions.count)
                    return ane_interop_io_copy_batched(
                        dst,
                        src,
                        dstBuf.baseAddress,
                        srcBuf.baseAddress,
                        chBuf.baseAddress,
                        count32,
                        spatial32
                    )
                }
            }
        }
        guard ok else { throw .interopCallFailed }
    }

    // MARK: - Lock/Unlock primitives

    /// Lock a surface for write access. Must be paired with `unlockWrite()`.
    @inline(__always)
    public static func lockWrite(_ surface: IOSurfaceRef) -> Bool {
        ane_interop_io_lock_write(surface)
    }

    /// Unlock a surface from write access.
    @inline(__always)
    @discardableResult
    public static func unlockWrite(_ surface: IOSurfaceRef) -> Bool {
        ane_interop_io_unlock_write(surface)
    }

    /// Lock a surface for read-only access. Must be paired with `unlockRead()`.
    @inline(__always)
    public static func lockRead(_ surface: IOSurfaceRef) -> Bool {
        ane_interop_io_lock_read(surface)
    }

    /// Unlock a surface from read-only access.
    @inline(__always)
    @discardableResult
    public static func unlockRead(_ surface: IOSurfaceRef) -> Bool {
        ane_interop_io_unlock_read(surface)
    }

    // MARK: - Unlocked I/O (caller must hold appropriate lock)

    /// Write FP16 data to a surface that is already locked for write.
    public static func writeFP16Unlocked(to surface: IOSurfaceRef, data: UnsafeBufferPointer<Float>, channels: Int, spatial: Int) {
        let count = checkedElementCount(channels: channels, spatial: spatial)
        precondition(data.count == count)
        if count == 0 { return }
        guard let src = data.baseAddress else {
            preconditionFailure("Input base address is nil for non-empty buffer")
        }
        let channels32 = Int32(clamping: channels)
        let spatial32 = Int32(clamping: spatial)
        precondition(Int(channels32) == channels && Int(spatial32) == spatial)
        let ok = ane_interop_io_write_fp16_unlocked(surface, src, channels32, spatial32)
        precondition(ok)
    }

    /// Read FP16 data from a surface that is already locked for read.
    public static func readFP16Unlocked(from surface: IOSurfaceRef,
                                        into dst: UnsafeMutableBufferPointer<Float>,
                                        channelOffset: Int,
                                        channels: Int,
                                        spatial: Int) {
        let count = checkedElementCount(channels: channels, spatial: spatial)
        precondition(channelOffset >= 0)
        precondition(dst.count == count)
        if count == 0 { return }
        guard let out = dst.baseAddress else {
            preconditionFailure("Destination base address is nil for non-empty buffer")
        }
        let channelOffset32 = Int32(clamping: channelOffset)
        let channels32 = Int32(clamping: channels)
        let spatial32 = Int32(clamping: spatial)
        precondition(Int(channelOffset32) == channelOffset)
        precondition(Int(channels32) == channels && Int(spatial32) == spatial)
        let ok = ane_interop_io_read_fp16_unlocked(surface, channelOffset32, out, channels32, spatial32)
        precondition(ok)
    }

    public static func copyFP16FromMultipleSources(dst: IOSurfaceRef,
                                                   spatial: Int,
                                                   regions: [FP16SourceCopyRegion]) throws(SurfaceIOError) {
        let spatial32 = try checkedNonNegativeInt32(spatial)
        if regions.isEmpty || spatial == 0 { return }

        var sources = [Unmanaged<IOSurfaceRef>?]()
        var dstOffsets = [Int32]()
        var srcOffsets = [Int32]()
        var channelCounts = [Int32]()
        sources.reserveCapacity(regions.count)
        dstOffsets.reserveCapacity(regions.count)
        srcOffsets.reserveCapacity(regions.count)
        channelCounts.reserveCapacity(regions.count)

        for region in regions {
            let dstOffset32 = try checkedNonNegativeInt32(region.dstChannelOffset)
            let srcOffset32 = try checkedNonNegativeInt32(region.srcChannelOffset)
            let channels32 = try checkedNonNegativeInt32(region.channels)
            _ = checkedElementCount(channels: region.channels, spatial: spatial)
            sources.append(Unmanaged.passUnretained(region.source))
            dstOffsets.append(dstOffset32)
            srcOffsets.append(srcOffset32)
            channelCounts.append(channels32)
        }

        let ok = sources.withUnsafeBufferPointer { srcBuf in
            dstOffsets.withUnsafeMutableBufferPointer { dstBuf in
                srcOffsets.withUnsafeMutableBufferPointer { srcOffBuf in
                    channelCounts.withUnsafeMutableBufferPointer { chBuf in
                        let count32 = Int32(clamping: regions.count)
                        precondition(Int(count32) == regions.count)
                        return ane_interop_io_copy_multi_src(
                            dst,
                            srcBuf.baseAddress,
                            dstBuf.baseAddress,
                            srcOffBuf.baseAddress,
                            chBuf.baseAddress,
                            count32,
                            spatial32
                        )
                    }
                }
            }
        }
        guard ok else { throw .interopCallFailed }
    }
}
