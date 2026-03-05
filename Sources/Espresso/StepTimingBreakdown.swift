import Darwin

public struct StepTimingBreakdown {
    public var tAne: Double
    public var tIO: Double
    public var tCls: Double
    public var tElem: Double
    public var tRms: Double
    public var tCblasWait: Double

    public init(
        tAne: Double = 0,
        tIO: Double = 0,
        tCls: Double = 0,
        tElem: Double = 0,
        tRms: Double = 0,
        tCblasWait: Double = 0
    ) {
        self.tAne = tAne
        self.tIO = tIO
        self.tCls = tCls
        self.tElem = tElem
        self.tRms = tRms
        self.tCblasWait = tCblasWait
    }
}

enum RuntimeClock {
    private static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    @inline(__always)
    static func now() -> UInt64 {
        mach_absolute_time()
    }

    @inline(__always)
    static func ms(_ delta: UInt64) -> Double {
        let nanos = (Double(delta) * Double(timebase.numer)) / Double(timebase.denom)
        return nanos / 1_000_000.0
    }

    @inline(__always)
    static func us(_ delta: UInt64) -> Double {
        let nanos = (Double(delta) * Double(timebase.numer)) / Double(timebase.denom)
        return nanos / 1_000.0
    }
}
