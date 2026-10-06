#pragma once
#include "strata/core/conversation_cache.hpp"
#include <filesystem>
#include <memory>
#include <string>

namespace strata::core {

// Ephemeral, process-exclusive SSD parking. Files are never imported across
// engine restarts: model weights, RoPE, steering and executable identity cannot
// accidentally share a cache. Exact tokens/images and cvec still select a branch.
class ConversationDiskCapture {
public:
    ~ConversationDiskCapture();
    ConversationStorageFactory factory();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

class ConversationDiskCache {
public:
    using Match = ConversationCache::Match;
    ConversationDiskCache(const std::string& directory, uint64_t budget, size_t slots);
    ~ConversationDiskCache();
    bool enabled() const;
    uint64_t bytes() const;
    size_t size() const;
    size_t evictions() const;
    const std::string& error() const;
    Match best(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;
    Match best(const std::vector<int64_t>& ids, const std::vector<ConversationImageKey>& images, bool cvec) const;
    void pin(size_t id);
    void unpin();
    void discard(size_t id);
    std::unique_ptr<ConversationDiskCapture> begin(uint64_t estimate, std::string& error);
    // Checkpoints are borrowed from the live session, not cloned for parking.
    // Overrides contain first stage followed by later stages (no nesting).
    bool put(ConversationDiskCapture& capture, const SavedConversation& image,
             const std::vector<const std::vector<ConversationCheckpoint>*>& checkpoints, std::string& error);
    // All checksums are checked with bounded staging before any GPU write. The
    // caller parks its outgoing state first, then frees old CPU checkpoints.
    bool verify(size_t id, std::string& error);
    bool load(size_t id, SavedConversation& image, std::string& error);
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace strata::core
