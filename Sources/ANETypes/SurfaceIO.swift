import IOSurface
import ANEInterop

public enum SurfaceIOError: Error, Equatable {
    case argumentOutOfRange
    case interopCallFailed
}

public enum SurfaceIO {
    @inline(__always)
    private static func checkedElementCount(channels: Int, spatial: Int) -> Int {
        precondition(channels >= 0 && spatial >= 0)
        let count = channels.multipliedReportingOverflow(by: spatial)
        precondition(!count.overflow)
        return count.partialValue
    }

    @inline(__always)
    private static func checkedOffsetElements(channelOffset: Int, spatial: Int) -> Int {
        precondition(channelOffset >= 0 && spatial >= 0)
        let elems = channelOffset.multipliedReportingOverflow(by: spatial)
        precondition(!elems.overflow)
        return elems.partialValue
    }

    @inline(__always)
    private static func checkedFP16Bytes(_ elementCount: Int) -> Int {
        let bytes = elementCount.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
        precondition(!bytes.overflow)
        return bytes.partialValue
    }

    @inline(__always)
    private static func ensureSurfaceRange(_ surface: IOSurfaceRef, byteOffset: Int, byteCount: Int) {
        precondition(byteOffset >= 0 && byteCount >= 0)
        let end = byteOffset.addingReportingOverflow(byteCount)
        precondition(!end.overflow)
        precondition(end.partialValue <= IOSurfaceGetAllocSize(surface))
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

        precondition(IOSurfaceLock(surface, [], nil) == kIOReturnSuccess)
        defer { IOSurfaceUnlock(surface, [], nil) }

        let base = IOSurfaceGetBaseAddress(surface)
        ensureSurfaceRange(surface, byteOffset: 0, byteCount: checkedFP16Bytes(count))

        precondition(count <= Int(Int32.max))
        guard let src = data.baseAddress else {
            preconditionFailure("Input base address is nil for non-empty buffer")
        }
        ane_interop_cvt_f32_to_f16(base, src, Int32(count))
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

        let offsetElems = checkedOffsetElements(channelOffset: channelOffset, spatial: spatial)
        let offsetBytes = checkedFP16Bytes(offsetElems)
        let payloadBytes = checkedFP16Bytes(count)
        ensureSurfaceRange(surface, byteOffset: offsetBytes, byteCount: payloadBytes)

        precondition(IOSurfaceLock(surface, .readOnly, nil) == kIOReturnSuccess)
        defer { IOSurfaceUnlock(surface, .readOnly, nil) }

        let base = IOSurfaceGetBaseAddress(surface)
        guard let out = dst.baseAddress else {
            preconditionFailure("Destination base address is nil for non-empty buffer")
        }
        precondition(count <= Int(Int32.max))
        let src = UnsafeRawPointer(base).advanced(by: offsetBytes)
        ane_interop_cvt_f16_to_f32(out, src, Int32(count))
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
}
