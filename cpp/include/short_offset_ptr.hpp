#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

#include <boost/functional/hash.hpp>

#include <int48_t.hpp>

template <typename T>
class short_offset_ptr final {
    public:
        using pointer       = typename std::add_pointer<T>::type;
        using const_pointer = typename std::add_const<pointer>::type;

        // destructor
        ~short_offset_ptr() = default;


        // short_offset_ptr interaction
        short_offset_ptr(const short_offset_ptr<T>& obj) : _offset(p2o(obj.get())) {}
        short_offset_ptr(short_offset_ptr<T>&& obj) noexcept : _offset(p2o(obj.get())) {}

        short_offset_ptr<T>& operator=(const short_offset_ptr<T>& obj) {
            _offset = p2o(obj.get());
            return *this;
        }

        short_offset_ptr<T>& operator=(short_offset_ptr<T>&& obj) noexcept {
            _offset = p2o(obj.get());
            return *this;
        }

        bool operator==(const short_offset_ptr<T>& obj) const {
            return o2p(_offset) == obj.get();
        }

        bool operator!=(const short_offset_ptr<T>& obj) const {
            return o2p(_offset) != obj.get();
        }


        // nullptr interaction
        short_offset_ptr() : _offset(O_NULL) {}
        short_offset_ptr(std::nullptr_t) : _offset(O_NULL) {}
        short_offset_ptr& operator=(std::nullptr_t) {
            _offset = O_NULL;
            return *this;
        }

        bool operator==(std::nullptr_t) const {
            return _offset == O_NULL;
        }

        bool operator!=(std::nullptr_t) const {
            return _offset != O_NULL;
        }


        // T* interaction
        short_offset_ptr(const_pointer p) : _offset(p2o(p)) {}
        short_offset_ptr<T>& operator=(const_pointer p) {
            _offset = p2o(p);
            return *this;
        }

        bool operator==(const_pointer p) const {
            return p == o2p(_offset);
        }

        bool operator!=(const_pointer p) const {
            return p != o2p(_offset);
        }


        // pointer operators
        const T& operator*() const {
            return *o2p(_offset);
        }

        T& operator*() {
            return *const_cast<pointer>(o2p(_offset));
        }

        const_pointer operator->() const {
            return o2p(_offset);
        }

        pointer operator->() {
            return const_cast<pointer>(o2p(_offset));
        }


        // bool interaction
        operator bool() const {
            return _offset != O_NULL;
        }


        // direct pointer/state access
        const_pointer get() const {
            return o2p(_offset);
        }

        pointer get() {
            return const_cast<T*>(o2p(_offset));
        }

        const int48_t& offset() const {
            return _offset;
        }

        int48_t offset() {
            return _offset;
        }

    private:
        static constexpr std::int64_t O_NULL = 1;

        union converter_p {
            pointer p;
            std::uint64_t u;
        };

        union converter_t {
            short_offset_ptr<T>* p;
            std::uint64_t              u;
        };

        int48_t _offset;

        int48_t p2o(const_pointer p) const {
            if (p == nullptr) {
                return O_NULL;
            }

            converter_p cp;
            cp.p = p;

            converter_t ct;
            ct.p = const_cast<short_offset_ptr<T>*>(this);

            return static_cast<std::int64_t>(cp.u) - static_cast<std::int64_t>(ct.u);
        }

        const_pointer o2p(const int48_t& o) const {
            if (o == O_NULL) {
                return nullptr;
            }

            converter_t ct;
            ct.p = const_cast<short_offset_ptr<T>*>(this);

            converter_p cp;
            cp.u = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(ct.u) + static_cast<std::int64_t>(o)
            );

            return cp.p;
        }
};

namespace std {
    template <typename T>
    struct hash<short_offset_ptr<T>> {
        size_t operator()(const short_offset_ptr<T>& obj) const {
            // XXX: implement fast path
            return _helper(static_cast<const T*>(obj));
        }

        std::hash<const T*> _helper;
    };
}

template <typename T>
struct offset_hash {
    explicit offset_hash(const short_offset_ptr<std::uint8_t> anchor_) : anchor(anchor_) {}

    size_t operator()(const short_offset_ptr<T>& obj) const {
        union {
            const std::uint8_t* p;
            std::int64_t i;
        } converter_anchor;

        union {
            const void* p;
            std::int64_t i;
        } converter_obj;

        converter_anchor.p = anchor.get();
        converter_obj.p = obj.get();

        std::size_t seed = 0;
        boost::hash_combine(seed, converter_anchor.i - converter_obj.i);
        return seed;
    }

    const short_offset_ptr<std::uint8_t> anchor;
};
