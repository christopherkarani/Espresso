import Foundation
import IOSurface
import Testing
import ANEGraphIR
@testable import LoRAAdapter

private func borrowedAllocation(_ adapter: borrowing LoRAAdapter) -> Int {
    adapter.allocationByteCount
}

private func makeBlob(payload: [UInt16]) -> Data {
    var data = Data(count: 128 + payload.count * MemoryLayout<UInt16>.stride)
    data.withUnsafeMutableBytes { raw in
        let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
        base[0] = 1
        base[4] = 2
        base[64] = 0xEF
        base[65] = 0xBE
        base[66] = 0xAD
        base[67] = 0xDE
        base[68] = 1
        raw.storeBytes(of: UInt32(payload.count * MemoryLayout<UInt16>.stride).littleEndian, toByteOffset: 72, as: UInt32.self)
        raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)

        let payloadPtr = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
        for (index, value) in payload.enumerated() {
            payloadPtr[index] = value
        }
    }
    return data
}

private func readSurfacePrefix(_ surface: IOSurfaceRef, count: Int) -> [UInt16] {
    let status = IOSurfaceLock(surface, .readOnly, nil)
    precondition(status == kIOReturnSuccess)
    defer { IOSurfaceUnlock(surface, .readOnly, nil) }

    let baseAddress = IOSurfaceGetBaseAddress(surface).assumingMemoryBound(to: UInt16.self)
    return Array(UnsafeBufferPointer(start: baseAddress, count: count))
}

private func fileStem(for target: LoRAConfig.LoRATarget) -> String {
    switch target {
    case .q: return "q"
    case .k: return "k"
    case .v: return "v"
    case .o: return "o"
    case .w1: return "w1"
    case .w2: return "w2"
    case .w3: return "w3"
    }
}

private func writeAdapterFixture(
    root: URL,
    layers: Int,
    targets: [LoRAConfig.LoRATarget],
    aElementCount: Int,
    bElementCount: Int,
    valueSeed: UInt16
) throws {
    let fm = FileManager.default
    for layer in 0..<layers {
        let layerDir = root.appendingPathComponent("layer\(layer)", isDirectory: true)
        try fm.createDirectory(at: layerDir, withIntermediateDirectories: true)

        for (targetIndex, target) in targets.enumerated() {
            let start = valueSeed &+ UInt16(layer * 32 + targetIndex * 4)
            let aPayload = (0..<aElementCount).map { start &+ UInt16($0) }
            let bPayload = (0..<bElementCount).map { start &+ 100 &+ UInt16($0) }
            try makeBlob(payload: aPayload).write(to: layerDir.appendingPathComponent("\(fileStem(for: target))_A.bin"))
            try makeBlob(payload: bPayload).write(to: layerDir.appendingPathComponent("\(fileStem(for: target))_B.bin"))
        }
    }
}

@Test func loraConfigStoresRankAlphaAndTargets() {
    let config = LoRAConfig(rank: 16, alpha: 32, targets: [.q, .v, .w1])

    #expect(config.rank == 16)
    #expect(config.alpha == 32)
    #expect(config.targets == [.q, .v, .w1])
}

@Test func loraAdapterCompilesInBorrowingContexts() {
    let adapter = LoRAAdapter(config: LoRAConfig(rank: 8, alpha: 16, targets: [.q]), nLayers: 1, inDim: 4, outDim: 6)
    #expect(borrowedAllocation(adapter) >= 49_152)
}

@Test func adapterSurfacesExposeExpectedLogicalShapes() {
    let adapter = LoRAAdapter(
        config: LoRAConfig(rank: 8, alpha: 16, targets: [.q, .w1]),
        nLayers: 2,
        inDim: 12,
        outDim: 20
    )

    let shapes = adapter.logicalShapes(forLayer: 1, target: .q)
    let expectedA = try! ANEShape(batch: 1, channels: 1, height: 12, spatial: 8)
    let expectedB = try! ANEShape(batch: 1, channels: 1, height: 20, spatial: 8)
    #expect(shapes.A == expectedA)
    #expect(shapes.B == expectedB)
}

@Test func allAdapterSurfacesUseUniformAllocationSizes() {
    let adapter = LoRAAdapter(
        config: LoRAConfig(rank: 16, alpha: 32, targets: [.q, .k, .v]),
        nLayers: 3,
        inDim: 24,
        outDim: 40
    )
    let expectedBytes = max(49_152, max(24, 40) * 16 * MemoryLayout<UInt16>.stride)

    #expect(adapter.allocationByteCount == expectedBytes)

    for layer in 0..<3 {
        for target in adapter.orderedTargetsForTesting() {
            let surfaces = adapter.surfaces(forLayer: layer, target: target)
            #expect(IOSurfaceGetAllocSize(surfaces.A) == expectedBytes)
            #expect(IOSurfaceGetAllocSize(surfaces.B) == expectedBytes)
        }
    }
}

@Test func loadAndSwapReplaceSurfacePayloadsWithoutReallocation() throws {
    let config = LoRAConfig(rank: 8, alpha: 16, targets: [.q, .w1])
    let adapter = LoRAAdapter(config: config, nLayers: 2, inDim: 6, outDim: 10)
    let targets = adapter.orderedTargetsForTesting()
    let aElementCount = 6 * config.rank
    let bElementCount = 10 * config.rank

    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let replacement = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: replacement)
    }

    try writeAdapterFixture(
        root: root,
        layers: 2,
        targets: targets,
        aElementCount: aElementCount,
        bElementCount: bElementCount,
        valueSeed: 3
    )
    try writeAdapterFixture(
        root: replacement,
        layers: 2,
        targets: targets,
        aElementCount: aElementCount,
        bElementCount: bElementCount,
        valueSeed: 50
    )

    let beforeAlloc = IOSurfaceGetAllocSize(adapter.surfaces(forLayer: 0, target: .q).A)

    try adapter.load(from: root)
    let firstAPrefix = readSurfacePrefix(adapter.surfaces(forLayer: 0, target: .q).A, count: 4)
    let firstBPrefix = readSurfacePrefix(adapter.surfaces(forLayer: 1, target: .w1).B, count: 4)
    #expect(firstAPrefix == [3, 4, 5, 6])
    #expect(firstBPrefix == [139, 140, 141, 142])

    try adapter.swap(from: replacement)
    let swappedAPrefix = readSurfacePrefix(adapter.surfaces(forLayer: 0, target: .q).A, count: 4)
    let swappedBPrefix = readSurfacePrefix(adapter.surfaces(forLayer: 1, target: .w1).B, count: 4)
    #expect(swappedAPrefix == [50, 51, 52, 53])
    #expect(swappedBPrefix == [186, 187, 188, 189])
    #expect(IOSurfaceGetAllocSize(adapter.surfaces(forLayer: 0, target: .q).A) == beforeAlloc)
}
