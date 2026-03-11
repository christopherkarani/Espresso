import Foundation
import ObjectiveC.runtime

/// ObjC runtime introspection helper for private ANE classes.
///
/// This is intentionally in EspressoBench so we can quickly iterate on reverse-engineering
/// without polluting the runtime modules.
enum ANEIntrospector {
    static func dump(className: String, filter: String?) -> String {
        guard let cls = NSClassFromString(className) else {
            return "=== \(className) ===\nNOT FOUND\n"
        }

        var out = "=== \(className) ===\n"
        out += dumpMethods(for: object_getClass(cls), header: "Class Methods", filter: filter)
        out += dumpMethods(for: cls, header: "Instance Methods", filter: filter)
        out += dumpProperties(for: cls, filter: filter)
        return out
    }

    private static func dumpMethods(for cls: AnyClass?, header: String, filter: String?) -> String {
        guard let cls else { return "" }
        var count: UInt32 = 0
        guard let methods = class_copyMethodList(cls, &count), count > 0 else { return "" }
        defer { free(methods) }

        var lines: [String] = []
        lines.reserveCapacity(Int(count))
        for i in 0..<Int(count) {
            let m = methods[i]
            let sel = method_getName(m)
            let name = NSStringFromSelector(sel)
            if let filter, !name.localizedCaseInsensitiveContains(filter) { continue }
            let enc = method_getTypeEncoding(m).map { String(cString: $0) } ?? "?"
            lines.append("  \(name)  [\(enc)]")
        }
        guard !lines.isEmpty else { return "" }
        return "\(header):\n" + lines.joined(separator: "\n") + "\n"
    }

    private static func dumpProperties(for cls: AnyClass, filter: String?) -> String {
        var count: UInt32 = 0
        guard let props = class_copyPropertyList(cls, &count), count > 0 else { return "" }
        defer { free(props) }

        var lines: [String] = []
        lines.reserveCapacity(Int(count))
        for i in 0..<Int(count) {
            let p = props[i]
            let pname = String(cString: property_getName(p))
            if let filter, !pname.localizedCaseInsensitiveContains(filter) { continue }
            let attrs = property_getAttributes(p).map { String(cString: $0) } ?? "?"
            lines.append("  @property \(pname)  [\(attrs)]")
        }
        guard !lines.isEmpty else { return "" }
        return "Properties:\n" + lines.joined(separator: "\n") + "\n"
    }
}
