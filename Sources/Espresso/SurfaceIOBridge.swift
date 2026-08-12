import ANERuntime
import ANETypes

/// Bridges untyped `rethrows` boundaries (`withUnsafeBufferPointer` and friends)
/// into `throws(ANEError)` contexts. The stdlib buffer accessors erase typed
/// throws to `any Error`, so SurfaceIO failures must be re-mapped at the boundary.
@inline(__always)
func mapSurfaceIOToANEError<R>(_ body: () throws -> R) throws(ANEError) -> R {
    do {
        return try body()
    } catch let error as SurfaceIOError {
        throw .surfaceIO(error)
    } catch let error as ANEError {
        throw error
    } catch {
        throw .surfaceIO(.interopCallFailed)
    }
}

@inline(__always)
func mapSurfaceIOToGenerationError<R>(_ body: () throws -> R) throws(GenerationError) -> R {
    do {
        return try body()
    } catch let error as SurfaceIOError {
        throw .runtimeFailure("SurfaceIO: \(error)")
    } catch let error as GenerationError {
        throw error
    } catch {
        throw .runtimeFailure("SurfaceIO interop failure")
    }
}

@inline(__always)
func mapSurfaceIOToMetalAttentionError<R>(_ body: () throws -> R) throws(MetalAttentionError) -> R {
    do {
        return try body()
    } catch let error as SurfaceIOError {
        throw .surfaceIOFailed
    } catch let error as MetalAttentionError {
        throw error
    } catch {
        throw .surfaceIOFailed
    }
}
