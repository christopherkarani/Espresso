import Darwin
import Foundation
import MachO

/// Exec-restart when ANE compile budget is exhausted.
/// Maps to `train_large.m:322-333`.
public enum ExecRestart {
    public static func message(step: Int, compileCount: Int, loss: Float) -> String {
        String(
            format: "[exec() restart step %d, %d compiles, loss=%.4f]",
            locale: Locale(identifier: "en_US_POSIX"),
            step,
            compileCount,
            loss
        )
    }

    /// Prints restart message, flushes output, and replaces the process image with `--resume`.
    /// Note: exec() does not run Swift deinits or ARC cleanup.
    public static func restart(step: Int, compileCount: Int, loss: Float) -> Never {
        print(message(step: step, compileCount: compileCount, loss: loss))
        fflush(stdout)
        fflush(stderr)

        let execPath = resolvedExecutablePath()
        let argv = restartArgv(currentArguments: CommandLine.arguments, resolvedExecPath: execPath)
        restart(execPath: execPath, argv: argv)
    }

    /// Resolve the current executable path for a robust exec restart.
    ///
    /// `CommandLine.arguments[0]` is not guaranteed to contain a `/` (e.g. when launched via `PATH`),
    /// which makes `execv` unreliable. dyld can provide the true executable location.
    internal static func resolvedExecutablePath() -> String {
        func dyldPath() -> String? {
            var bufSize = UInt32(MAXPATHLEN)
            var buf = [UInt8](repeating: 0, count: Int(bufSize))

            func tryFill(_ buf: inout [UInt8], _ size: inout UInt32) -> Int32 {
                buf.withUnsafeMutableBufferPointer { ptr in
                    guard let base = ptr.baseAddress else { return -1 }
                    return base.withMemoryRebound(to: CChar.self, capacity: ptr.count) { cbuf in
                        _NSGetExecutablePath(cbuf, &size)
                    }
                }
            }

            var result = tryFill(&buf, &bufSize)
            if result != 0 {
                // `bufSize` now contains the required size.
                guard bufSize > 0 else { return nil }
                buf = [UInt8](repeating: 0, count: Int(bufSize))
                result = tryFill(&buf, &bufSize)
            }
            guard result == 0 else { return nil }

            let raw = buf.withUnsafeBufferPointer { ptr -> String in
                ptr.baseAddress!.withMemoryRebound(to: CChar.self, capacity: ptr.count) { cbuf in
                    String(cString: cbuf)
                }
            }

            // Canonicalize to an absolute path when possible.
            return raw.withCString { cstr in
                guard let rp = realpath(cstr, nil) else { return raw }
                defer { free(rp) }
                return String(cString: rp)
            }
        }

        if let resolved = dyldPath(), !resolved.isEmpty {
            return resolved
        }

        // Fallback: best-effort (may be a bare name; `restart(execPath:argv:)` uses execvp).
        return CommandLine.arguments.first ?? ""
    }

    /// Build argv for exec-restart while preserving all original CLI flags, and ensuring `--resume`
    /// is present exactly once.
    internal static func restartArgv(currentArguments: [String], resolvedExecPath: String) -> [String] {
        precondition(!resolvedExecPath.isEmpty)
        precondition(!currentArguments.isEmpty)

        var rest: [String] = []
        rest.reserveCapacity(max(0, currentArguments.count - 1) + 1)

        var sawResume = false
        for arg in currentArguments.dropFirst() {
            if arg == "--resume" {
                sawResume = true
                continue
            }
            rest.append(arg)
        }
        if !sawResume {
            rest.append("--resume")
        } else {
            // Normalize to exactly once (move to the end).
            rest.append("--resume")
        }

        return [resolvedExecPath] + rest
    }

    /// Replace the current process image with the given argv.
    ///
    /// ObjC uses `execl`, but `execl` is unavailable from Swift due to varargs import rules.
    /// `execv` provides the same exec-restart semantics (replaces process image; no deinits run).
    public static func restart(execPath: String, argv: [String]) -> Never {
        precondition(!execPath.isEmpty)
        precondition(!argv.isEmpty)

        func withCStrings<R>(_ strings: [String], _ body: ([UnsafePointer<CChar>]) -> R) -> R {
            var ptrs: [UnsafePointer<CChar>] = []
            ptrs.reserveCapacity(strings.count)

            func loop(_ index: Int) -> R {
                if index == strings.count {
                    return body(ptrs)
                }
                return strings[index].withCString { cstr in
                    ptrs.append(cstr)
                    return loop(index + 1)
                }
            }
            return loop(0)
        }

	        withCStrings(argv) { ptrs in
	            let cargv = UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>.allocate(capacity: ptrs.count + 1)
	            for i in 0..<ptrs.count {
	                cargv[i] = UnsafeMutablePointer(mutating: ptrs[i])
	            }
	            cargv[ptrs.count] = nil
	
	            _ = execPath.withCString { cpath in
	                // execvp searches PATH when `execPath` does not contain '/', matching typical shell semantics.
	                execvp(cpath, cargv)
	            }
	
	            // If we got here, execv failed.
	            cargv.deallocate()
	        }
	
	        perror("execvp")
	        exit(1)
    }
}
