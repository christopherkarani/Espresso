import Testing
@testable import ModelSupport

@Test func qwenInstructPromptWrapsOfficialChatTemplate() {
    let wrapped = QwenInstructPrompt.wrapUserTurn("The capital of France is")
    #expect(
        wrapped ==
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
    )
}

@Test func qwenInstructPromptShouldWrapFollowsModelFamily() {
    let qwen = MultiModelConfig(
        name: "Qwen2.5-0.5B-Instruct",
        nLayer: 24,
        nHead: 14,
        nKVHead: 2,
        dModel: 896,
        headDim: 64,
        hiddenDim: 4864,
        vocab: 151_936,
        maxSeq: 4096,
        normEps: 1e-6,
        architecture: .llama
    )
    #expect(QwenInstructPrompt.shouldWrap(config: qwen))
    #expect(!QwenInstructPrompt.shouldWrap(config: ModelRegistry.stories110m))
    #expect(!QwenInstructPrompt.shouldWrap(config: ModelRegistry.llama3_2_1b))
}
