import IOSurface
import ANEInterop

public enum SurfaceIOError: Error, Equatable {
    case argumentOutOfRange
    case interopCallFailed
}

public enum SurfaceIO {
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
