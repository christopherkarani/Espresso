// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Espresso",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "espresso-train", targets: ["EspressoTrain"]),
        .executable(name: "espresso-bench", targets: ["EspressoBench"]),
        .executable(name: "espresso-multitoken-probe", targets: ["EspressoMultitokenProbe"]),
        .library(name: "Espresso", targets: ["Espresso"]),
    ],
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
            dependencies: ["ANETypes"],
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
            path: "Sources/EspressoMultitokenProbe",
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
        .testTarget(name: "CPUOpsTests", dependencies: ["CPUOps"], swiftSettings: [.swiftLanguageMode(.v6)]),
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
        .target(
            name: "DeltaCompilation",
            dependencies: ["ANEInterop", "ANETypes", "ANERuntime"],
            path: "Sources/DeltaCompilation",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .target(
            name: "LoRAAdapter",
            dependencies: ["ANEGraphIR", "ANEBuilder", "ANETypes"],
            path: "Sources/LoRAAdapter",
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("IOSurface")]
        ),
        .testTarget(
            name: "ANEBuilderTests",
            dependencies: ["ANEBuilder", "ANEGraphIR"],
            path: "Tests/ANEBuilderTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "DeltaCompilationTests",
            dependencies: ["DeltaCompilation", "ANEGraphIR"],
            path: "Tests/DeltaCompilationTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
        .testTarget(
            name: "LoRAAdapterTests",
            dependencies: ["LoRAAdapter", "ANEGraphIR"],
            path: "Tests/LoRAAdapterTests",
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
