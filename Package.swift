// swift-tools-version: 6.2
import PackageDescription

// Zero third-party package dependencies. Apple system frameworks only.
let package = Package(
    name: "Espresso",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "espresso-train", targets: ["EspressoTrain"]),
        .executable(name: "espresso-bench", targets: ["EspressoBench"]),
        .executable(name: "espresso-generate", targets: ["EspressoGenerate"]),
        .executable(name: "espresso-multitoken-probe", targets: ["EspressoMultitokenProbe"]),
        .executable(name: "espc", targets: ["ESPCompilerCLI"]),
        .executable(name: "esprun", targets: ["ESPRuntimeCLI"]),
        .library(name: "ANETypes", targets: ["ANETypes"]),
        .library(name: "ANERuntime", targets: ["ANERuntime"]),
        .library(name: "Espresso", targets: ["Espresso"]),
        .library(name: "ESPBundle", targets: ["ESPBundle"]),
        .library(name: "ESPCompiler", targets: ["ESPCompiler"]),
        .library(name: "ESPRuntime", targets: ["ESPRuntime"]),
        .library(name: "ESPConvert", targets: ["ESPConvert"]),
        .library(name: "ESPBenchSupport", targets: ["ESPBenchSupport"]),
        .library(name: "ModelSupport", targets: ["ModelSupport"]),
        .library(name: "RealModelInference", targets: ["RealModelInference"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "ANEInterop",
            path: "Sources/ANEInterop",
            publicHeadersPath: "include",
            cSettings: [
                .unsafeFlags(["-fobjc-arc", "-O2"]),
                .headerSearchPath("include"),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("CoreML"),
                .linkedFramework("IOSurface"),
                .linkedLibrary("dl"),
            ]
        ),
        .target(
            name: "ANETypes",
            dependencies: ["ANEInterop"],
            path: "Sources/ANETypes",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("IOSurface")]
        ),
        .target(
            name: "MILGenerator",
            dependencies: ["ANETypes", "ANEGraphIR", "ANEBuilder", "ANECodegen", "ANEPasses"],
            path: "Sources/MILGenerator",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "CPUOps",
            dependencies: ["ANETypes"],
            path: "Sources/CPUOps",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "ANERuntime",
            dependencies: ["ANEInterop", "ANETypes", "MILGenerator"],
            path: "Sources/ANERuntime",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("IOSurface")]
        ),
        .target(
            name: "Espresso",
            dependencies: ["ANERuntime", "CPUOps", "ANETypes"],
            path: "Sources/Espresso",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreML"),
                .linkedFramework("IOSurface"),
                .linkedFramework("Metal"),
            ]
        ),
        .executableTarget(
            name: "EspressoTrain",
            dependencies: ["Espresso"],
            path: "Sources/EspressoTrain",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "EspressoBench",
            dependencies: ["Espresso", "ANERuntime", "ANETypes", "CPUOps", "MILGenerator"],
            path: "Sources/EspressoBench",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("IOSurface"),
                .linkedFramework("CoreML"),
            ]
        ),
        .executableTarget(
            name: "EspressoMultitokenProbe",
            dependencies: ["Espresso", "ANERuntime", "ANETypes"],
            path: "Sources/EspressoMultiTokenProbe",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("IOSurface"),
            ]
        ),
        .testTarget(name: "ANEInteropTests", dependencies: ["ANEInterop"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(name: "ANETypesTests", dependencies: ["ANETypes"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(
            name: "MILGeneratorTests",
            dependencies: ["MILGenerator", "ANERuntime"],
            path: "Tests/MILGeneratorTests",
            resources: [.process("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(name: "CPUOpsTests", dependencies: ["CPUOps", "ANETypes"], swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(
            name: "ANERuntimeTests",
            dependencies: ["ANERuntime", "ANEInterop", "MILGenerator", "ANETypes", "Espresso"],
            path: "Tests/ANERuntimeTests",
            resources: [.copy("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "EspressoTests",
            dependencies: ["Espresso", "CPUOps", "ANEInterop", "ANETypes"],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "CrossValidationTests",
            dependencies: ["ANERuntime", "CPUOps", "ANETypes", "Espresso", "MILGenerator"],
            path: "Tests/CrossValidationTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ANEGraphIR",
            path: "Sources/ANEGraphIR",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ANEGraphIRTests",
            dependencies: ["ANEGraphIR"],
            path: "Tests/ANEGraphIRTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ANECodegen",
            dependencies: ["ANEGraphIR"],
            path: "Sources/ANECodegen",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ANECodegenTests",
            dependencies: ["ANECodegen", "ANEGraphIR"],
            path: "Tests/ANECodegenTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ANEPasses",
            dependencies: ["ANEGraphIR"],
            path: "Sources/ANEPasses",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ANEPassesTests",
            dependencies: ["ANEPasses", "ANEGraphIR"],
            path: "Tests/ANEPassesTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ANEBuilder",
            dependencies: ["ANEGraphIR"],
            path: "Sources/ANEBuilder",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ANEBuilderTests",
            dependencies: ["ANEBuilder", "ANEGraphIR"],
            path: "Tests/ANEBuilderTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ModelSupport",
            dependencies: ["ANEGraphIR", "ANEBuilder", "ANECodegen", "ANEPasses", "ANETypes"],
            path: "Sources/ModelSupport",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ModelSupportTests",
            dependencies: ["ModelSupport", "ANEGraphIR", "ANETypes", "ANEPasses"],
            path: "Tests/ModelSupportTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "MigrationParityTests",
            dependencies: ["MILGenerator", "ANETypes", "ANEGraphIR", "ANEBuilder", "ANECodegen", "ANEPasses"],
            path: "Tests/MigrationParityTests",
            resources: [.process("Fixtures")],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "RealModelInference",
            dependencies: ["ModelSupport", "ANEGraphIR", "ANEBuilder", "ANECodegen", "ANEPasses", "ANERuntime", "ANETypes", "ANEInterop", "CPUOps", "Espresso"],
            path: "Sources/RealModelInference",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("IOSurface"),
            ]
        ),
        .executableTarget(
            name: "EspressoGenerate",
            dependencies: [
                "RealModelInference", "ModelSupport", "ANETypes", "ANERuntime", "ESPBundle", "ESPRuntime",
                "ESPBenchSupport",
            ],
            path: "Sources/EspressoGenerate",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "EspressoQwenParity",
            dependencies: ["RealModelInference", "ModelSupport"],
            path: "Sources/EspressoQwenParity",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "EspressoGenerateTests",
            dependencies: ["EspressoGenerate", "ModelSupport", "ANETypes", "ESPBundle", "ESPBenchSupport"],
            path: "Tests/EspressoGenerateTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "RealModelInferenceTests",
            dependencies: [
                "RealModelInference", "ModelSupport", "ANEGraphIR", "ANETypes", "Espresso",
            ],
            path: "Tests/RealModelInferenceTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ESPBundle",
            dependencies: [],
            path: "Sources/ESPBundle",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("Foundation")]
        ),
        .testTarget(
            name: "ESPBundleTests",
            dependencies: ["ESPBundle"],
            path: "Tests/ESPBundleTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ESPCompiler",
            dependencies: ["ESPBundle", "ModelSupport", "ANETypes"],
            path: "Sources/ESPCompiler",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ESPCompilerTests",
            dependencies: ["ESPCompiler", "ESPBundle"],
            path: "Tests/ESPCompilerTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ESPRuntime",
            dependencies: ["ESPBundle", "ESPCompiler", "ModelSupport", "RealModelInference"],
            path: "Sources/ESPRuntime",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ESPRuntimeTests",
            dependencies: ["ESPRuntime", "ESPBundle"],
            path: "Tests/ESPRuntimeTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ESPCompilerCLITests",
            dependencies: ["ESPCompilerCLI"],
            path: "Tests/ESPCompilerCLITests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ESPConvert",
            dependencies: ["ESPBundle", "ESPCompiler", "ModelSupport"],
            path: "Sources/ESPConvert",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ESPConvertTests",
            dependencies: ["ESPConvert", "ESPBundle", "ESPCompiler"],
            path: "Tests/ESPConvertTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "ESPBenchSupport",
            dependencies: ["ESPBundle"],
            path: "Sources/ESPBenchSupport",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "ESPBenchSupportTests",
            dependencies: ["ESPBenchSupport", "ESPBundle"],
            path: "Tests/ESPBenchSupportTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "ESPCompilerCLI",
            dependencies: ["ESPBundle", "ESPConvert"],
            path: "Sources/ESPCompilerCLI",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .executableTarget(
            name: "ESPRuntimeCLI",
            dependencies: ["ESPBundle", "ESPRuntime"],
            path: "Sources/ESPRuntimeCLI",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
