// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TrainingLoop",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/christopherkarani/Espresso.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "TrainingLoop",
            dependencies: [
                .product(name: "Espresso",     package: "Espresso"),
                .product(name: "ANERuntime",   package: "Espresso"),
                .product(name: "ANETypes",     package: "Espresso"),
                .product(name: "ModelSupport", package: "Espresso"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
