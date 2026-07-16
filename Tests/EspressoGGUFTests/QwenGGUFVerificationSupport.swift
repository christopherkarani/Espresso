import EdgeRunnerIO

enum QwenGGUFVerificationSupport {
    static func rawGGUFLMHeadTensorName(from weightMap: WeightMap) -> String? {
        if weightMap["output.weight"] != nil {
            return "output.weight"
        }
        if weightMap["token_embd.weight"] != nil {
            return "token_embd.weight"
        }
        return nil
    }
}
