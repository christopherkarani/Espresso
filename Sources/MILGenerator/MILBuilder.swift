import Foundation

struct MILBuilder {
    static let posixLocale = Locale(identifier: "en_US_POSIX")

    private(set) var text: String = ""

    init(reserveCapacity: Int = 0) {
        if reserveCapacity > 0 {
            text.reserveCapacity(reserveCapacity)
        }
    }

    mutating func append(_ s: String) {
        text += s
    }

    mutating func appendLine(_ s: String) {
        text += s
        text += "\n"
    }

    mutating func appendFP16(_ value: Float) {
        // ObjC generators use `%f` which defaults to 6 fractional digits.
        text += String(format: "%.6f", locale: Self.posixLocale, Double(value))
    }
}

enum MILText {
    static let header =
        """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
        """
        + "\n"

    static let convConst =
        """
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
        """
        + "\n"
}
