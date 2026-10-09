#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace astervec {

// A caller-chosen point id: a 64-bit unsigned integer or a string (UUIDs are
// passed as their string form). The integer 7 and the string "7" are
// different points. Strings in canonical UUID form are stored in 16 bytes
// and read back as the identical string.
//
// In a store created with IdMode::kKeyed the engine maps every PointId to a
// dense internal id itself; in a kDense store only integers up to kMaxRealId
// are accepted and the integer is the internal id.
class PointId {
public:
    static constexpr size_t kMaxStringBytes = 1024;

    PointId() = default;  // the integer 0
    static PointId Int(uint64_t v) { PointId p; p.int_ = v; return p; }
    static PointId Str(std::string s)
    {
        PointId p;
        p.is_int_ = false;
        p.str_ = std::move(s);
        return p;
    }

    bool is_int() const { return is_int_; }
    uint64_t as_int() const { return int_; }
    const std::string& as_str() const { return str_; }

    bool operator==(const PointId& o) const
    {
        return is_int_ == o.is_int_ && (is_int_ ? int_ == o.int_ : str_ == o.str_);
    }
    bool operator!=(const PointId& o) const { return !(*this == o); }

    // Stored form: a type tag, then the value. Integers are 8 bytes
    // big-endian so the stored order matches the numeric order. A string in
    // canonical UUID form (36 chars, lowercase hex, hyphens at 8/13/18/23 --
    // what str(uuid.UUID) produces) is stored as its 16 bytes; it decodes
    // back to exactly the same string, so callers cannot tell the
    // difference. Any other string is stored as is.
    std::string Encode() const
    {
        std::string out;
        if (!is_int_ && IsCanonicalUuid(str_)) {
            out.resize(17);
            out[0] = kUuidTag;
            size_t o = 1;
            int high = -1;  // first hex digit of the current byte, if any
            for (char c : str_) {
                if (c == '-') continue;
                const unsigned v = HexVal(c);
                if (high < 0) {
                    high = static_cast<int>(v);
                } else {
                    out[o++] = static_cast<char>((static_cast<unsigned>(high) << 4) | v);
                    high = -1;
                }
            }
        } else if (is_int_) {
            out.resize(9);
            out[0] = kIntTag;
            uint64_t v = int_;
            for (int i = 8; i >= 1; --i) {
                out[static_cast<size_t>(i)] = static_cast<char>(v & 0xFFu);
                v >>= 8;
            }
        } else {
            out.reserve(1 + str_.size());
            out.push_back(kStrTag);
            out += str_;
        }
        return out;
    }

    static bool Decode(std::string_view bytes, PointId* out)
    {
        if (bytes.empty()) return false;
        if (bytes[0] == kIntTag) {
            if (bytes.size() != 9) return false;
            uint64_t v = 0;
            for (size_t i = 1; i < 9; ++i) {
                v = (v << 8) | static_cast<unsigned char>(bytes[i]);
            }
            *out = Int(v);
            return true;
        }
        if (bytes[0] == kStrTag) {
            *out = Str(std::string(bytes.substr(1)));
            return true;
        }
        if (bytes[0] == kUuidTag) {
            if (bytes.size() != 17) return false;
            static constexpr char kHex[] = "0123456789abcdef";
            std::string s;
            s.reserve(36);
            for (size_t i = 1; i < 17; ++i) {
                if (i == 5 || i == 7 || i == 9 || i == 11) s.push_back('-');
                const auto b = static_cast<unsigned char>(bytes[i]);
                s.push_back(kHex[b >> 4]);
                s.push_back(kHex[b & 0x0F]);
            }
            *out = Str(std::move(s));
            return true;
        }
        return false;
    }

    static bool IsCanonicalUuid(std::string_view s)
    {
        if (s.size() != 36) return false;
        for (size_t i = 0; i < 36; ++i) {
            const char c = s[i];
            if (i == 8 || i == 13 || i == 18 || i == 23) {
                if (c != '-') return false;
            } else if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
                return false;
            }
        }
        return true;
    }

private:
    static constexpr char kIntTag = 0x01;
    static constexpr char kStrTag = 0x02;
    static constexpr char kUuidTag = 0x03;

    static unsigned HexVal(char c)
    {
        return c <= '9' ? static_cast<unsigned>(c - '0')
                        : static_cast<unsigned>(c - 'a' + 10);
    }

    bool is_int_ = true;
    uint64_t int_ = 0;
    std::string str_;
};

}  // namespace astervec
