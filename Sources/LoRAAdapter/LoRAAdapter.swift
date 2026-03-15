import Foundation
import IOSurface
import ANEGraphIR

public struct LoRAConfig: Sendable {
    public enum LoRATarget: Sendable, Hashable, CaseIterable {
        case q, k, v, o, w1, w2, w3
    }

    public let rank: Int
    public let alpha: Float
    public let targets: Set<LoRATarget>

    public init(rank: Int, alpha: Float, targets: Set<LoRATarget>) {
        self.rank = rank
        self.alpha = alpha
        self.targets = targets
    }
}

enum LoRAAdapterError: Error, CustomStringConvertible {
    case missingBlobFile(String)
    case malformedBlobFile(String)
    case unexpectedPayloadSize(path: String, expected: Int, actual: Int)
    case surfaceLockFailed(String, IOReturn)
    case surfaceTooSmall(path: String, expected: Int, actual: Int)

    var description: String {
        switch self {
        case let .missingBlobFile(path):
            return "Missing LoRA blob file: \(path)"
        case let .malformedBlobFile(path):
            return "Malformed LoRA blob file: \(path)"
        case let .unexpectedPayloadSize(path, expected, actual):
            return "LoRA blob payload size mismatch for \(path): expected \(expected), got \(actual)"
        case let .surfaceLockFailed(path, status):
            return "Failed to lock IOSurface for \(path): \(status)"
        case let .surfaceTooSmall(path, expected, actual):
            return "IOSurface too small for \(path): expected \(expected), got \(actual)"
        }
    }
}

public struct LoRAAdapter: ~Copyable {
    private struct SurfacePair {
        let aSurface: IOSurfaceRef
        let bSurface: IOSurfaceRef
        let aShape: ANEShape
        let bShape: ANEShape
        let aLogicalBytes: Int
        let bLogicalBytes: Int
    }

    private static let minimumAllocationBytes = 49_152
    private static let payloadSizeFieldOffset = 72
    private static let payloadOffsetFieldOffset = 80
    private static let headerSize = 128

    public let config: LoRAConfig

    private let nLayers: Int
    private let orderedTargets: [LoRAConfig.LoRATarget]
    private let uniformAllocationBytes: Int
    private let slots: [SurfacePair]

    public init(config: LoRAConfig, nLayers: Int, inDim: Int, outDim: Int) {
        precondition(nLayers > 0, "nLayers must be positive")
        precondition(inDim > 0, "inDim must be positive")
        precondition(outDim > 0, "outDim must be positive")
        precondition([8, 16, 32].contains(config.rank), "LoRA rank must be one of 8, 16, 32")
        precondition(!config.targets.isEmpty, "LoRA targets must be non-empty")

        let orderedTargets = LoRAConfig.LoRATarget.allCases.filter(config.targets.contains)
        let allocationBytes = Self.uniformAllocationBytes(inDim: inDim, outDim: outDim, rank: config.rank)
        let aShape = try! ANEShape(batch: 1, channels: 1, height: inDim, spatial: config.rank)
        let bShape = try! ANEShape(batch: 1, channels: 1, height: outDim, spatial: config.rank)
        let aLogicalBytes = aShape.byteSize(for: .fp16)
        let bLogicalBytes = bShape.byteSize(for: .fp16)

        self.config = config
        self.nLayers = nLayers
        self.orderedTargets = orderedTargets
        self.uniformAllocationBytes = allocationBytes

        var pairs: [SurfacePair] = []
        pairs.reserveCapacity(nLayers * orderedTargets.count)

        for _ in 0..<nLayers {
            for _ in orderedTargets {
                pairs.append(
                    SurfacePair(
                        aSurface: Self.makeSurface(bytes: allocationBytes),
                        bSurface: Self.makeSurface(bytes: allocationBytes),
                        aShape: aShape,
                        bShape: bShape,
                        aLogicalBytes: aLogicalBytes,
                        bLogicalBytes: bLogicalBytes
                    )
                )
            }
        }

        self.slots = pairs
    }

    public func load(from directory: URL) throws {
        try loadOrSwap(from: directory)
    }

    public func load(from directory: String) throws {
        try load(from: URL(fileURLWithPath: directory))
    }

    public func swap(from directory: URL) throws {
        try loadOrSwap(from: directory)
    }

    public func swap(from directory: String) throws {
        try swap(from: URL(fileURLWithPath: directory))
    }

    public func surfaces(forLayer layer: Int, target: LoRAConfig.LoRATarget) -> (A: IOSurfaceRef, B: IOSurfaceRef) {
        let slot = slots[slotIndex(layer: layer, target: target)]
        return (slot.aSurface, slot.bSurface)
    }

    internal var allocationByteCount: Int {
        uniformAllocationBytes
    }

    internal func orderedTargetsForTesting() -> [LoRAConfig.LoRATarget] {
        orderedTargets
    }

    internal func logicalShapes(forLayer layer: Int, target: LoRAConfig.LoRATarget) -> (A: ANEShape, B: ANEShape) {
        let slot = slots[slotIndex(layer: layer, target: target)]
        return (slot.aShape, slot.bShape)
    }

    private static func uniformAllocationBytes(inDim: Int, outDim: Int, rank: Int) -> Int {
        let maxDimProduct = max(inDim, outDim).multipliedReportingOverflow(by: rank)
        precondition(!maxDimProduct.overflow, "LoRA shape overflow")
        let byteCount = maxDimProduct.partialValue.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
        precondition(!byteCount.overflow, "LoRA byte size overflow")
        return max(Self.minimumAllocationBytes, byteCount.partialValue)
    }

    private static func makeSurface(bytes: Int) -> IOSurfaceRef {
        let properties: [CFString: Any] = [
            kIOSurfaceWidth: bytes,
            kIOSurfaceHeight: 1,
            kIOSurfaceBytesPerElement: 1,
            kIOSurfaceBytesPerRow: bytes,
            kIOSurfaceAllocSize: bytes,
            kIOSurfacePixelFormat: 0,
        ]
        guard let surface = IOSurfaceCreate(properties as CFDictionary) else {
            fatalError("Failed to allocate LoRA IOSurface")
        }
        return surface
    }

    private func loadOrSwap(from directory: URL) throws {
        for layer in 0..<nLayers {
            let layerDirectory = directory.appendingPathComponent("layer\(layer)", isDirectory: true)
            for target in orderedTargets {
                let slot = slots[slotIndex(layer: layer, target: target)]
                try Self.writeBlob(
                    at: layerDirectory.appendingPathComponent("\(target.fileStem)_A.bin"),
                    logicalBytes: slot.aLogicalBytes,
                    allocationBytes: uniformAllocationBytes,
                    to: slot.aSurface
                )
                try Self.writeBlob(
                    at: layerDirectory.appendingPathComponent("\(target.fileStem)_B.bin"),
                    logicalBytes: slot.bLogicalBytes,
                    allocationBytes: uniformAllocationBytes,
                    to: slot.bSurface
                )
            }
        }
    }

    private func slotIndex(layer: Int, target: LoRAConfig.LoRATarget) -> Int {
        precondition(layer >= 0 && layer < nLayers, "Layer index out of range")
        guard let targetIndex = orderedTargets.firstIndex(of: target) else {
            preconditionFailure("Target \(target) not configured")
        }
        return layer * orderedTargets.count + targetIndex
    }

    private static func writeBlob(
        at url: URL,
        logicalBytes: Int,
        allocationBytes: Int,
        to surface: IOSurfaceRef
    ) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoRAAdapterError.missingBlobFile(url.path)
        }

        let blob = try Data(contentsOf: url)
        let (payloadOffset, payloadSize) = try parseBlobHeader(blob, path: url.path)
        guard payloadSize == logicalBytes else {
            throw LoRAAdapterError.unexpectedPayloadSize(
                path: url.path,
                expected: logicalBytes,
                actual: payloadSize
            )
        }

        let availableBytes = IOSurfaceGetAllocSize(surface)
        guard allocationBytes <= availableBytes else {
            throw LoRAAdapterError.surfaceTooSmall(
                path: url.path,
                expected: allocationBytes,
                actual: availableBytes
            )
        }

        let lockStatus = IOSurfaceLock(surface, [], nil)
        guard lockStatus == kIOReturnSuccess else {
            throw LoRAAdapterError.surfaceLockFailed(url.path, lockStatus)
        }
        defer { IOSurfaceUnlock(surface, [], nil) }

        let baseAddress = IOSurfaceGetBaseAddress(surface)
        memset(baseAddress, 0, allocationBytes)
        _ = blob.withUnsafeBytes { raw in
            memcpy(baseAddress, raw.baseAddress!.advanced(by: payloadOffset), payloadSize)
        }
    }

    private static func parseBlobHeader(_ blob: Data, path: String) throws -> (offset: Int, size: Int) {
        guard blob.count >= headerSize else {
            throw LoRAAdapterError.malformedBlobFile(path)
        }

        let payloadSize: UInt32 = blob.withUnsafeBytes { raw in
            raw.load(fromByteOffset: payloadSizeFieldOffset, as: UInt32.self)
        }
        let payloadOffset: UInt32 = blob.withUnsafeBytes { raw in
            raw.load(fromByteOffset: payloadOffsetFieldOffset, as: UInt32.self)
        }

        let size = Int(UInt32(littleEndian: payloadSize))
        let offset = Int(UInt32(littleEndian: payloadOffset))
        guard size >= 0,
              offset >= Self.headerSize,
              size.isMultiple(of: MemoryLayout<UInt16>.stride),
              offset <= blob.count,
              size <= blob.count - offset
        else {
            throw LoRAAdapterError.malformedBlobFile(path)
        }
        return (offset, size)
    }
}

extension LoRAConfig.LoRATarget {
    var fileStem: String {
        switch self {
        case .q: return "q"
        case .k: return "k"
        case .v: return "v"
        case .o: return "o"
        case .w1: return "w1"
        case .w2: return "w2"
        case .w3: return "w3"
        }
    }
}
