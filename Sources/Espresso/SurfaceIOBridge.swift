import ANERuntime
import ANETypes

// `mapSurfaceIOToANEError` lives in ANERuntime (public, beside the kernel sets
// that own surface bindings); Espresso call sites bind to it directly.

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
