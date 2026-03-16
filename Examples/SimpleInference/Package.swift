// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SimpleInference",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/christopherkarani/Espresso.git", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "SimpleInference",
            dependencies: [
                .product(name: "RealModelInference", package: "Espresso"),
                .product(name: "ModelSupport",       package: "Espresso"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
