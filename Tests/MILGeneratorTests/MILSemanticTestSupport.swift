import Foundation
import XCTest
import ANECodegen

func loadMILFixtureText(
    _ filename: String,
    file: StaticString = #filePath,
    line: UInt = #line
) throws -> String {
    let name = (filename as NSString).deletingPathExtension
    let ext = (filename as NSString).pathExtension
    let url =
        Bundle.module.url(forResource: name, withExtension: ext)
        ?? Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures")
    guard let url else {
        XCTFail("Fixture not found in test bundle: \(filename)", file: file, line: line)
        throw XCTSkip("Fixture missing: \(filename)")
    }
    return try String(contentsOf: url, encoding: .utf8)
}

func extractMILBlobPaths(_ mil: String) -> [String] {
    let regex = try! NSRegularExpression(pattern: #"BLOBFILE\(path=string\("([^"]+)"\)"#)
    let range = NSRange(mil.startIndex..<mil.endIndex, in: mil)
    return regex.matches(in: mil, range: range).compactMap { match in
        guard let range = Range(match.range(at: 1), in: mil) else { return nil }
        return String(mil[range])
    }
}

func extractMILReturnTuple(_ mil: String) -> [String] {
    guard let line = mil.split(whereSeparator: \.isNewline).first(where: { $0.contains("} -> (") }) else { return [] }
    guard let start = line.firstIndex(of: "("), let end = line.lastIndex(of: ")"), start < end else { return [] }
    return line[line.index(after: start)..<end]
        .split(separator: ",")
        .map { $0.replacingOccurrences(of: " ", with: "") }
}

func extractMILInputSignature(_ mil: String) -> String {
    guard let line = mil.split(whereSeparator: \.isNewline).first(where: { $0.contains("func main<ios18>(") }) else { return "" }
    return line.replacingOccurrences(of: " ", with: "")
}

func extractMILInputNames(_ mil: String) -> [String] {
    guard let line = mil.split(whereSeparator: \.isNewline).first(where: { $0.contains("func main<ios18>(") }) else { return [] }
    let signature = String(line)
    let regex = try! NSRegularExpression(pattern: #"tensor<[^>]+>\s+([A-Za-z_][A-Za-z0-9_]*)"#)
    let range = NSRange(signature.startIndex..<signature.endIndex, in: signature)
    return regex.matches(in: signature, range: range).compactMap { match in
        guard let range = Range(match.range(at: 1), in: signature) else { return nil }
        return String(signature[range])
    }
}

func assertMILSemanticParity(
    actual: String,
    fixture filename: String,
    file: StaticString = #filePath,
    line: UInt = #line
) throws {
    let expected = try loadMILFixtureText(filename, file: file, line: line)
    XCTAssertTrue(
        MILDiff.structuralEquiv(expected, actual),
        "Op sequence mismatch for fixture \(filename): \(MILDiff.extractOps(expected)) vs \(MILDiff.extractOps(actual))",
        file: file,
        line: line
    )
    XCTAssertEqual(extractMILInputSignature(actual), extractMILInputSignature(expected), file: file, line: line)
    XCTAssertEqual(extractMILReturnTuple(actual), extractMILReturnTuple(expected), file: file, line: line)
    for path in extractMILBlobPaths(expected) {
        XCTAssertTrue(actual.contains(path), "Missing weight path \(path) in \(filename)", file: file, line: line)
    }
}
