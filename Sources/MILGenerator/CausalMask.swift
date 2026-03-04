import Foundation
import ANETypes

public enum CausalMask {
    private static let cachedSeqLen = ModelConfig.seqLen
    private static let cachedBlob: Data = build(seqLen: ModelConfig.seqLen)
    nonisolated(unsafe) private static let cache: NSCache<NSNumber, NSData> = {
        let cache = NSCache<NSNumber, NSData>()
        cache.setObject(cachedBlob as NSData, forKey: NSNumber(value: cachedSeqLen))
        return cache
    }()

    public static func blob(seqLen: Int) -> Data {
        precondition(seqLen > 0)
        let key = NSNumber(value: seqLen)

        if let cached = cache.object(forKey: key) {
            return cached as Data
        }
        let generated = build(seqLen: seqLen)
        cache.setObject(generated as NSData, forKey: key)
        return generated
    }

    private static func build(seqLen: Int) -> Data {
        precondition(seqLen > 0)

        let minFP16: Float = -65504.0
        var mask = [Float](repeating: 0.0, count: seqLen * seqLen)
        for row in 0..<seqLen {
            for col in (row + 1)..<seqLen {
                mask[row * seqLen + col] = minFP16
            }
        }

        return WeightBlob.build(from: mask, rows: seqLen, cols: seqLen)
    }
}
