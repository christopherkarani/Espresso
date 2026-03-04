public struct AdamState: ~Copyable {
    public let m: TensorBuffer
    public let v: TensorBuffer
    public let count: Int

    public init(count: Int) {
        self.count = count
        self.m = TensorBuffer(count: count, zeroed: true)
        self.v = TensorBuffer(count: count, zeroed: true)
    }
}

