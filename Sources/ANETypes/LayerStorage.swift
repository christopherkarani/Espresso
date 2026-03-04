public struct LayerStorage<Element: ~Copyable>: ~Copyable {
    private let storage: UnsafeMutableBufferPointer<Element>
    public let count: Int

    public init(count: Int, initializer: (Int) -> Element) {
        precondition(count >= 0)
        self.count = count
        self.storage = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0..<count {
            storage.baseAddress!.advanced(by: i).initialize(to: initializer(i))
        }
    }

    /// Initialize a LayerStorage where element construction may throw.
    ///
    /// This is required for storing `~Copyable` types whose initializers can fail
    /// (e.g. ANE kernels during compilation). On a thrown error, already-initialized
    /// elements are deinitialized and the backing allocation is freed before rethrow.
    public init(count: Int, throwingInitializer: (Int) throws -> Element) rethrows {
        precondition(count >= 0)
        let storage = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        var initializedCount = 0
        do {
            for i in 0..<count {
                try storage.baseAddress!.advanced(by: i).initialize(to: throwingInitializer(i))
                initializedCount += 1
            }
        } catch {
            storage.baseAddress?.deinitialize(count: initializedCount)
            storage.deallocate()
            throw error
        }

        // Only assign to self after initialization succeeds to avoid deinit running on a
        // partially-initialized storage during error unwinding.
        self.count = count
        self.storage = storage
    }

    public subscript(index: Int) -> Element {
        _read {
            yield storage[index]
        }
        _modify {
            yield &storage[index]
        }
    }

    deinit {
        storage.baseAddress?.deinitialize(count: count)
        storage.deallocate()
    }
}
