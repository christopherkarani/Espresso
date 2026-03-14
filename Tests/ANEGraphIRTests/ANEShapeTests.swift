import Testing
@testable import ANEGraphIR

@Test func elementCount() {
    let shape = ANEShape(batch: 1, channels: 768, height: 1, spatial: 256)
    #expect(shape.elementCount == 196_608)
}

@Test func byteSizeFP16() {
    let shape = ANEShape(channels: 768, spatial: 256)
    #expect(shape.byteSize(for: .fp16) == 393_216) // 196608 * 2
}

@Test func byteSizeFP32() {
    let shape = ANEShape(channels: 768, spatial: 256)
    #expect(shape.byteSize(for: .fp32) == 786_432) // 196608 * 4
}

@Test func dimensionsArray() {
    let shape = ANEShape(batch: 1, channels: 768, height: 1, spatial: 256)
    #expect(shape.dimensions == [1, 768, 1, 256])
}

@Test func defaultBatchAndHeight() {
    let shape = ANEShape(channels: 512, spatial: 64)
    #expect(shape.batch == 1)
    #expect(shape.height == 1)
    #expect(shape.channels == 512)
    #expect(shape.spatial == 64)
}

@Test func meetsMinimumIOSurfaceSize() {
    // 768 * 32 * 2 = 49,152 bytes — exactly at minimum
    let atMinimum = ANEShape(channels: 768, spatial: 32)
    #expect(atMinimum.meetsMinimumIOSurfaceSize(for: .fp16))

    // 768 * 16 * 2 = 24,576 bytes — below minimum
    let belowMinimum = ANEShape(channels: 768, spatial: 16)
    #expect(!belowMinimum.meetsMinimumIOSurfaceSize(for: .fp16))

    // 4 * 4 * 2 = 32 bytes — way below
    let tiny = ANEShape(channels: 4, spatial: 4)
    #expect(!tiny.meetsMinimumIOSurfaceSize(for: .fp16))
}

@Test func exceedsSRAMBudget() {
    // 32768 * 1024 * 2 = 67,108,864 bytes = 64MB > 32MB
    let large = ANEShape(channels: 32768, spatial: 1024)
    #expect(large.exceedsSRAMBudget(for: .fp16))

    // 768 * 256 * 2 = 393,216 bytes << 32MB
    let normal = ANEShape(channels: 768, spatial: 256)
    #expect(!normal.exceedsSRAMBudget(for: .fp16))
}

@Test func equalityCheck() {
    let a = ANEShape(channels: 768, spatial: 256)
    let b = ANEShape(channels: 768, spatial: 256)
    let c = ANEShape(channels: 768, spatial: 128)
    #expect(a == b)
    #expect(a != c)
}
