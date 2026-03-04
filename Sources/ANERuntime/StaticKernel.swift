import Foundation
import IOSurface
import MILGenerator

/// Wraps a weight-free sdpaBwd2 ANE kernel.
/// Compiled once at startup (or after exec-restart). NOT recompiled when weights change.
/// Different lifecycle from LayerKernelSet: survives LayerKernelSet recompilation.
/// One instance per layer.
public struct StaticKernel: ~Copyable {
    internal struct CompileContract: Equatable {
        internal let weightCount: Int
        internal let inputBytes: Int
        internal let outputBytes: Int
    }

    public let kernel: ANEKernel

    internal static var compileContract: CompileContract {
        let generator = SDPABackward2Generator()
        return CompileContract(
            weightCount: 0,
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }

    /// Compile a weight-free sdpaBwd2 kernel.
    public init() throws(ANEError) {
        let generator = SDPABackward2Generator()
        self.kernel = try ANEKernel(
            milText: generator.milText,
            weights: [],
            inputBytes: generator.inputBytes,
            outputBytes: generator.outputBytes
        )
    }
}
