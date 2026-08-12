import ANERuntime
import ANETypes

/// Bridges untyped `rethrows` boundaries (`withUnsafeBufferPointer` and friends)
/// into `throws(RealModelInferenceError)` contexts. The stdlib buffer accessors
/// erase typed throws to `any Error`, so SurfaceIO failures must be re-mapped.
@inline(__always)
func mapSurfaceIOToRealModelError<R>(_ body: () throws -> R) throws(RealModelInferenceError) -> R {
    do {
        return try body()
    } catch let error as SurfaceIOError {
        throw .runtimeFailure("SurfaceIO: \(error)")
    } catch let error as RealModelInferenceError {
        throw error
    } catch let error as ANEError {
        throw .runtimeFailure("ANE: \(error)")
    } catch {
        throw .runtimeFailure("SurfaceIO interop failure")
    }
}
