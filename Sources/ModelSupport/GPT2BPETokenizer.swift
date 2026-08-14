import Foundation

public enum GPT2BPETokenizerError: Error, Sendable, Equatable {
    case malformedVocabulary
    case emptyVocabulary
    case nonContiguousTokenIDs(expectedCount: Int, actualMaxTokenID: Int)
    case malformedTokenizerJSON
    case unsupportedTokenizerModel(String)
    case invalidPreTokenizerPattern(String)
}

public final class GPT2BPETokenizer: Tokenizer, @unchecked Sendable {
    private static let defaultPreTokenizerPattern =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+"

    public let vocabSize: Int

    private let encoder: [String: Int]
    private let decoder: [String]
    private let bpeRanks: [TokenPair: Int]
    private let byteToUnicode: [UnicodeScalar]
    private let unicodeToByte: [UInt32: UInt8]
    private let preTokenizer: NSRegularExpression
    private let prefixTokenIDs: [Int]
    private let skippedTokenIDs: Set<Int>
    private let specialTokens: [(content: String, id: Int)]
    private let cacheLock = NSLock()
    private var cache: [String: [String]] = [:]

    public convenience init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        let rawObject = try JSONSerialization.jsonObject(with: vocabData)
        guard let rawVocab = rawObject as? [String: Any] else {
            throw GPT2BPETokenizerError.malformedVocabulary
        }

        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
        try self.init(
            encoder: try Self.parseVocabulary(rawVocab),
            merges: merges.split(separator: "\n").map(String.init),
            preTokenizerPattern: Self.defaultPreTokenizerPattern,
            prefixTokenIDs: [],
            skippedTokenIDs: [],
            specialTokens: []
        )
    }

    public convenience init(tokenizerJSONURL: URL) throws {
        let data = try Data(contentsOf: tokenizerJSONURL)
        let rawObject = try JSONSerialization.jsonObject(with: data)
        guard let root = rawObject as? [String: Any],
              let model = root["model"] as? [String: Any] else {
            throw GPT2BPETokenizerError.malformedTokenizerJSON
        }

        let modelType = model["type"] as? String ?? "unknown"
        guard modelType == "BPE" else {
            throw GPT2BPETokenizerError.unsupportedTokenizerModel(modelType)
        }

        guard let rawVocab = model["vocab"] as? [String: Any] else {
            throw GPT2BPETokenizerError.malformedTokenizerJSON
        }

        var encoder = try Self.parseVocabulary(rawVocab)
        let added = Self.mergeAddedTokens(from: root, into: &encoder)
        let merges = try Self.parseTokenizerJSONMerges(model["merges"])
        let preTokenizerPattern = Self.preTokenizerPattern(from: root) ?? Self.defaultPreTokenizerPattern
        let prefixTokenIDs = Self.prefixTokenIDs(from: root, encoder: encoder)

        try self.init(
            encoder: encoder,
            merges: merges,
            preTokenizerPattern: preTokenizerPattern,
            prefixTokenIDs: prefixTokenIDs,
            skippedTokenIDs: added.skippedIDs,
            specialTokens: added.specials
        )
    }

    public func encode(_ text: String) -> [Int] {
        var tokens = prefixTokenIDs
        if specialTokens.isEmpty {
            tokens.append(contentsOf: encodeOrdinary(text))
            return tokens
        }

        var start = text.startIndex
        while start < text.endIndex {
            if let match = specialPrefix(in: text, from: start) {
                tokens.append(match.id)
                start = match.end
                continue
            }
            var cursor = text.index(after: start)
            while cursor < text.endIndex, specialPrefix(in: text, from: cursor) == nil {
                cursor = text.index(after: cursor)
            }
            tokens.append(contentsOf: encodeOrdinary(String(text[start..<cursor])))
            start = cursor
        }
        return tokens
    }

    private func encodeOrdinary(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        let nsText = text as NSString
        let matches = preTokenizer.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        var tokens: [Int] = []
        for match in matches {
            let chunk = nsText.substring(with: match.range)
            let encoded = encodeBytes(of: chunk)
            for piece in bpe(encoded) {
                guard let token = encoder[piece] else { continue }
                tokens.append(token)
            }
        }
        return tokens
    }

    private func specialPrefix(in text: String, from start: String.Index) -> (id: Int, end: String.Index)? {
        for special in specialTokens {
            if text[start...].hasPrefix(special.content) {
                return (special.id, text.index(start, offsetBy: special.content.count))
            }
        }
        return nil
    }

    public func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for token in tokens where decoder.indices.contains(token) {
            if skippedTokenIDs.contains(token) {
                continue
            }
            let piece = decoder[token]
            for scalar in piece.unicodeScalars {
                if let byte = unicodeToByte[scalar.value] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(scalar).utf8)
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private init(
        encoder: [String: Int],
        merges: [String],
        preTokenizerPattern: String,
        prefixTokenIDs: [Int],
        skippedTokenIDs: Set<Int>,
        specialTokens: [(content: String, id: Int)]
    ) throws {
        guard !encoder.isEmpty else {
            throw GPT2BPETokenizerError.emptyVocabulary
        }
        self.encoder = encoder

        let maxTokenID = encoder.values.max() ?? -1
        guard maxTokenID >= 0 else {
            throw GPT2BPETokenizerError.emptyVocabulary
        }
        guard maxTokenID + 1 == encoder.count else {
            throw GPT2BPETokenizerError.nonContiguousTokenIDs(
                expectedCount: encoder.count,
                actualMaxTokenID: maxTokenID
            )
        }

        var decoder = Array(repeating: "", count: maxTokenID + 1)
        for (token, id) in encoder where decoder.indices.contains(id) {
            decoder[id] = token
        }
        self.decoder = decoder
        self.vocabSize = decoder.count
        self.prefixTokenIDs = prefixTokenIDs
        self.skippedTokenIDs = skippedTokenIDs
        self.specialTokens = specialTokens.sorted { $0.content.count > $1.content.count }

        var bpeRanks: [TokenPair: Int] = [:]
        var rank = 0
        for line in merges {
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            let pieces = line.split(separator: " ", omittingEmptySubsequences: true)
            guard pieces.count == 2 else { continue }
            bpeRanks[TokenPair(String(pieces[0]), String(pieces[1]))] = rank
            rank += 1
        }
        self.bpeRanks = bpeRanks

        let mapping = Self.makeByteMapping()
        self.byteToUnicode = mapping.byteToUnicode
        self.unicodeToByte = mapping.unicodeToByte
        do {
            self.preTokenizer = try NSRegularExpression(pattern: preTokenizerPattern)
        } catch {
            throw GPT2BPETokenizerError.invalidPreTokenizerPattern(preTokenizerPattern)
        }
    }

    private func encodeBytes(of text: String) -> String {
        var result = String.UnicodeScalarView()
        result.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            result.append(byteToUnicode[Int(byte)])
        }
        return String(result)
    }

    private func bpe(_ token: String) -> [String] {
        cacheLock.lock()
        if let cached = cache[token] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        var word = token.unicodeScalars.map { String($0) }
        guard word.count > 1 else {
            cacheLock.lock()
            cache[token] = word
            cacheLock.unlock()
            return word
        }

        while true {
            let pairs = Self.pairs(in: word)
            guard !pairs.isEmpty else { break }

            var bestPair: TokenPair?
            var bestRank = Int.max
            for pair in pairs {
                guard let rank = bpeRanks[pair], rank < bestRank else { continue }
                bestPair = pair
                bestRank = rank
            }

            guard let bestPair else { break }

            var merged: [String] = []
            var index = 0
            while index < word.count {
                if index < word.count - 1 &&
                    word[index] == bestPair.left &&
                    word[index + 1] == bestPair.right {
                    merged.append(bestPair.left + bestPair.right)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged
            if word.count == 1 {
                break
            }
        }

        cacheLock.lock()
        cache[token] = word
        cacheLock.unlock()
        return word
    }

    private static func pairs(in word: [String]) -> [TokenPair] {
        guard word.count > 1 else { return [] }
        return zip(word, word.dropFirst()).map(TokenPair.init)
    }

    private static func makeByteMapping() -> (byteToUnicode: [UnicodeScalar], unicodeToByte: [UInt32: UInt8]) {
        var direct: [UInt8] = []
        direct.append(contentsOf: 33...126)
        direct.append(contentsOf: 161...172)
        direct.append(contentsOf: 174...255)

        var byteToUnicode = Array(repeating: UnicodeScalar(0)!, count: 256)
        for byte in direct {
            byteToUnicode[Int(byte)] = UnicodeScalar(Int(byte))!
        }

        var nextScalar = 256
        for byte in 0..<256 where byteToUnicode[byte].value == 0 {
            byteToUnicode[byte] = UnicodeScalar(nextScalar)!
            nextScalar += 1
        }

        var unicodeToByte: [UInt32: UInt8] = [:]
        for (byte, scalar) in byteToUnicode.enumerated() {
            unicodeToByte[scalar.value] = UInt8(byte)
        }

        return (byteToUnicode, unicodeToByte)
    }

    private static func parseVocabulary(_ rawVocab: [String: Any]) throws -> [String: Int] {
        guard !rawVocab.isEmpty else {
            throw GPT2BPETokenizerError.emptyVocabulary
        }

        var encoder: [String: Int] = [:]
        encoder.reserveCapacity(rawVocab.count)
        for (token, value) in rawVocab {
            guard let number = value as? NSNumber else {
                throw GPT2BPETokenizerError.malformedVocabulary
            }
            encoder[token] = number.intValue
        }
        return encoder
    }

    private static func mergeAddedTokens(
        from root: [String: Any],
        into encoder: inout [String: Int]
    ) -> (skippedIDs: Set<Int>, specials: [(content: String, id: Int)]) {
        guard let addedTokens = root["added_tokens"] as? [[String: Any]] else {
            return ([], [])
        }
        var skippedTokenIDs = Set<Int>()
        var specials: [(content: String, id: Int)] = []
        for token in addedTokens {
            guard let id = (token["id"] as? NSNumber)?.intValue,
                  let content = token["content"] as? String else {
                continue
            }
            encoder[content] = id
            if (token["special"] as? Bool) == true {
                skippedTokenIDs.insert(id)
                specials.append((content, id))
            }
        }
        return (skippedTokenIDs, specials)
    }

    private static func parseTokenizerJSONMerges(_ rawMerges: Any?) throws -> [String] {
        guard let rawMerges else {
            throw GPT2BPETokenizerError.malformedTokenizerJSON
        }
        if let merges = rawMerges as? [String] {
            return merges
        }
        if let merges = rawMerges as? [[String]] {
            return merges.compactMap { merge in
                guard merge.count == 2 else { return nil }
                return "\(merge[0]) \(merge[1])"
            }
        }
        throw GPT2BPETokenizerError.malformedTokenizerJSON
    }

    private static func preTokenizerPattern(from root: [String: Any]) -> String? {
        if let preTokenizer = root["pre_tokenizer"] as? [String: Any] {
            return regexPattern(in: preTokenizer)
        }
        return nil
    }

    private static func regexPattern(in preTokenizer: [String: Any]) -> String? {
        if let pattern = (preTokenizer["pattern"] as? [String: Any])?["Regex"] as? String {
            return pattern
        }
        if let preTokenizers = preTokenizer["pretokenizers"] as? [[String: Any]] {
            for entry in preTokenizers {
                if let pattern = regexPattern(in: entry) {
                    return pattern
                }
            }
        }
        return nil
    }

    private static func prefixTokenIDs(from root: [String: Any], encoder: [String: Int]) -> [Int] {
        if let postProcessor = root["post_processor"] as? [String: Any] {
            if let prefix = prefixTokenIDs(in: postProcessor), !prefix.isEmpty {
                return prefix
            }
        }
        if let bosID = encoder["<|begin_of_text|>"] {
            return [bosID]
        }
        return []
    }

    private static func prefixTokenIDs(in processor: [String: Any]) -> [Int]? {
        if let processors = processor["processors"] as? [[String: Any]] {
            for entry in processors {
                if let prefix = prefixTokenIDs(in: entry) {
                    return prefix
                }
            }
        }

        if let specialTokens = processor["special_tokens"] as? [String: Any],
           let bos = specialTokens["<|begin_of_text|>"] as? [String: Any],
           let rawIDs = bos["ids"] as? [Any] {
            let ids = rawIDs.compactMap { ($0 as? NSNumber)?.intValue }
            if !ids.isEmpty {
                return ids
            }
        }

        guard let single = processor["single"] as? [[String: Any]] else {
            return nil
        }

        var prefix: [Int] = []
        for item in single {
            if let special = item["SpecialToken"] as? [String: Any],
               let rawIDs = special["ids"] as? [Any] {
                prefix.append(contentsOf: rawIDs.compactMap { ($0 as? NSNumber)?.intValue })
                continue
            }
            if item["Sequence"] != nil {
                break
            }
        }
        return prefix
    }
}

private struct TokenPair: Hashable {
    let left: String
    let right: String

    init(_ left: String, _ right: String) {
        self.left = left
        self.right = right
    }

    init(_ values: (String, String)) {
        self.init(values.0, values.1)
    }
}
