#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// Optional backing used only while capturing/restoring an SSD conversation.
// The engine's ordinary RAM snapshots keep their original segmented storage.
class ConversationStorage {
public:
    using Visitor = std::function<bool(uint8_t*, size_t, size_t)>;
    virtual ~ConversationStorage() = default;
    virtual bool visit(size_t offset, size_t count, bool write, const Visitor& fn) = 0;
};
using ConversationStorageFactory = std::function<std::shared_ptr<ConversationStorage>(size_t)>;
inline thread_local ConversationStorageFactory* conversation_storage_factory = nullptr;
class ConversationStorageScope {
    ConversationStorageFactory* previous_;
public:
    explicit ConversationStorageScope(ConversationStorageFactory& factory)
        : previous_(std::exchange(conversation_storage_factory, &factory)) {}
    ~ConversationStorageScope() { conversation_storage_factory = previous_; }
    ConversationStorageScope(const ConversationStorageScope&) = delete;
};

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)), backing_(std::move(other.backing_)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        backing_ = std::move(other.backing_);
        return *this;
    }

    size_t size() const { return size_; }
    bool external() const { return bool(backing_); }
    const std::shared_ptr<ConversationStorage>& backing() const { return backing_; }
    static ConversationBuffer backed(size_t n, std::shared_ptr<ConversationStorage> storage) {
        ConversationBuffer out; out.size_ = n; out.backing_ = std::move(storage); return out;
    }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (backing_) {
            if (n == size_) return;
            if (n == 0) { backing_.reset(); size_ = 0; return; }
            throw std::logic_error("SSD snapshot buffers cannot grow or retain mutable pages");
        }
        if (n && !size_ && conversation_storage_factory) {
            backing_ = (*conversation_storage_factory)(n);
            if (!backing_) throw std::runtime_error("SSD snapshot allocation failed");
            size_ = n;
            return; // Fresh capture writes every byte before publication.
        }
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        if (backing_ || other.backing_) {
            std::vector<uint8_t> left(std::min<size_t>(size_, 65536)), right(left.size());
            for (size_t at = 0; at < size_;) {
                const size_t n = std::min(left.size(), size_ - at);
                if (!read(left.data(), at, n) || !other.read(right.data(), at, n) ||
                    std::memcmp(left.data(), right.data(), n)) return false;
                at += n;
            }
            return true;
        }
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        if (self.backing_)
            return self.backing_->visit(offset, count, !std::is_const_v<Self>,
                [&](uint8_t* p, size_t n, size_t at) { return fn(p, n, at); });
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
    std::shared_ptr<ConversationStorage> backing_;
};

} // namespace strata::core
