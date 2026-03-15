import ANEGraphIR

public struct ANEOptimizationPipeline: Sendable {
    public static let maxIterations = 20

    @discardableResult
    public static func optimize(_ graph: inout ANEGraph) -> Bool {
        var didChangeGraph = false

        for _ in 0..<maxIterations {
            var iterationChanged = false

            var identityElimination = IdentityEliminationPass()
            iterationChanged = identityElimination.run(on: &graph) || iterationChanged

            var castElimination = CastEliminationPass()
            iterationChanged = castElimination.run(on: &graph) || iterationChanged

            var deadCodeElimination = DeadCodeEliminationPass()
            iterationChanged = deadCodeElimination.run(on: &graph) || iterationChanged

            didChangeGraph = didChangeGraph || iterationChanged

            if !iterationChanged {
                break
            }
        }

        return didChangeGraph
    }

    public static func validate(_ graph: ANEGraph) -> [ANEConstraint] {
        ANEValidationPass().run(on: graph)
    }
}
