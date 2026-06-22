#pragma once

#ifndef LSM_VEC_LOGGER_H_
#define LSM_VEC_LOGGER_H_

#include <atomic>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace astervec {

enum class LogSeverity : uint8_t {
    DEBUG = 0,
    INFO,
    WARN,
    ERR,
    FATAL
};

enum class LogChoice : uint8_t {
    NONE = 0,
    FILE,
    STDOUT,
    STDERR,
    CUSTOM
};

static constexpr std::size_t DEFAULT_LOG_BUFFER_SIZE = 2048;
static constexpr std::size_t TIME_STRING_SIZE = 64;

inline const char* severityToString(LogSeverity severity) {
    switch (severity) {
        case LogSeverity::DEBUG: return "DEBUG";
        case LogSeverity::INFO:  return "INFO";
        case LogSeverity::WARN:  return "WARN";
        case LogSeverity::ERR:   return "ERR";
        case LogSeverity::FATAL: return "FATAL";
        default:                 return "UNKNOWN";
    }
}

inline void getCurrentTimeString(char* outBuf, std::size_t outSize) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto nowTimeT = system_clock::to_time_t(now);

    std::tm tmBuf{};
#if defined(_WIN32)
    localtime_s(&tmBuf, &nowTimeT);
#else
    localtime_r(&nowTimeT, &tmBuf);
#endif

    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;

    std::snprintf(outBuf, outSize,
                  "%04d-%02d-%02d %02d:%02d:%02d.%03lld",
                  tmBuf.tm_year + 1900, tmBuf.tm_mon + 1, tmBuf.tm_mday,
                  tmBuf.tm_hour, tmBuf.tm_min, tmBuf.tm_sec,
                  static_cast<long long>(ms));
}

class Logger {
public:
    virtual ~Logger() = default;

    virtual void log(LogSeverity severity,
                     const char* file,
                     int line,
                     const char* function,
                     const char* msg,
                     std::size_t msgLen) = 0;
};

class NullLogger final : public Logger {
public:
    void log(LogSeverity, const char*, int, const char*, const char*, std::size_t) override {}
};

inline void writeLine(FILE* out,
                      LogSeverity severity,
                      const char* file,
                      int line,
                      const char* function,
                      const char* msg,
                      std::size_t msgLen)
{
    char timeBuf[TIME_STRING_SIZE];
    getCurrentTimeString(timeBuf, sizeof(timeBuf));

    char headBuf[512];
    const int headLen = std::snprintf(
        headBuf, sizeof(headBuf),
        "[%s] [%s] %s:%d (%s) ",
        timeBuf, severityToString(severity), file, line, function);

    if (headLen > 0) {
        std::fwrite(headBuf, 1, static_cast<std::size_t>(headLen), out);
    }
    if (msg && msgLen > 0) {
        std::fwrite(msg, 1, msgLen, out);
    }
    std::fwrite("\n", 1, 1, out);
}

class FileLogger final : public Logger {
public:
    explicit FileLogger(const char* logFilePath) {
        if (logFilePath == nullptr || logFilePath[0] == '\0') {
            throw std::runtime_error("FileLogger: logFilePath is null/empty.");
        }
        file_ = std::fopen(logFilePath, "ab");
        if (!file_) {
            throw std::runtime_error("FileLogger: failed to open log file.");
        }
    }

    ~FileLogger() override {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    void log(LogSeverity severity,
             const char* file,
             int line,
             const char* function,
             const char* msg,
             std::size_t msgLen) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        writeLine(file_, severity, file, line, function, msg, msgLen);
        std::fflush(file_);
    }

private:
    FILE* file_{nullptr};
    std::mutex mu_;
};

class StdStreamLogger final : public Logger {
public:
    explicit StdStreamLogger(FILE* file) : file_(file) {
        if (!file_) {
            throw std::runtime_error("StdStreamLogger: file is null.");
        }
    }

    void log(LogSeverity severity,
             const char* file,
             int line,
             const char* function,
             const char* msg,
             std::size_t msgLen) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        writeLine(file_, severity, file, line, function, msg, msgLen);
        std::fflush(file_);
    }

private:
    FILE* file_{nullptr};
    std::mutex mu_;
};

class LogBuffer {
public:
    LogBuffer() : size_(0) {
        data_.fill('\0');
    }

    const char* data() const { return data_.data(); }
    std::size_t size() const { return size_; }

    void clear() {
        size_ = 0;
        if (!data_.empty()) data_[0] = '\0';
    }

    void append(std::string_view sv) {
        if (sv.empty()) return;
        const std::size_t canCopy = remaining();
        const std::size_t toCopy = (sv.size() < canCopy) ? sv.size() : canCopy;
        if (toCopy == 0) return;
        std::memcpy(data_.data() + size_, sv.data(), toCopy);
        size_ += toCopy;
        if (size_ < data_.size()) data_[size_] = '\0';
    }

    void appendChar(char c) {
        if (remaining() == 0) return;
        data_[size_++] = c;
        if (size_ < data_.size()) data_[size_] = '\0';
    }

    template <typename IntT>
    void appendInt(IntT value) {
        static_assert(std::is_integral_v<IntT>, "appendInt requires integral type.");
        char tmp[64];
        auto res = std::to_chars(tmp, tmp + sizeof(tmp), value);
        if (res.ec == std::errc()) {
            append(std::string_view(tmp, static_cast<std::size_t>(res.ptr - tmp)));
        } else {
            append("[int_conv_err]");
        }
    }

    void appendDouble(double value) {
        char tmp[64];
        const int n = std::snprintf(tmp, sizeof(tmp), "%.6f", value);
        if (n > 0) {
            append(std::string_view(tmp, static_cast<std::size_t>(n)));
        } else {
            append("[float_conv_err]");
        }
    }

private:
    std::size_t remaining() const {
        if (data_.size() == 0) return 0;
        if (size_ >= data_.size() - 1) return 0;
        return (data_.size() - 1) - size_;
    }

private:
    std::array<char, DEFAULT_LOG_BUFFER_SIZE> data_;
    std::size_t size_;
};

class LogItem {
public:
    LogItem(LogSeverity severity,
            const char* file,
            int line,
            const char* function,
            Logger* logger)
        : severity_(severity),
          file_(file),
          line_(line),
          function_(function),
          logger_(logger)
    {}

    ~LogItem() {
        if (logger_) {
            logger_->log(severity_, file_, line_, function_, buffer_.data(), buffer_.size());
        }
        if (severity_ == LogSeverity::FATAL) {
            std::fflush(nullptr);
            std::abort();
        }
    }

    LogItem(const LogItem&) = delete;
    LogItem& operator=(const LogItem&) = delete;

    LogItem(LogItem&& other) noexcept
        : severity_(other.severity_),
          file_(other.file_),
          line_(other.line_),
          function_(other.function_),
          logger_(other.logger_)
    {
        buffer_ = other.buffer_;
        other.logger_ = nullptr;
    }

    LogItem& operator<<(const char* s) {
        if (s) buffer_.append(std::string_view(s));
        else buffer_.append("[null]");
        return *this;
    }

    LogItem& operator<<(std::string_view sv) {
        buffer_.append(sv);
        return *this;
    }

    LogItem& operator<<(char c) {
        buffer_.appendChar(c);
        return *this;
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    LogItem& operator<<(T v) {
        if constexpr (std::is_signed_v<T>) {
            buffer_.appendInt<long long>(static_cast<long long>(v));
        } else {
            buffer_.appendInt<unsigned long long>(static_cast<unsigned long long>(v));
        }
        return *this;
    }

    LogItem& operator<<(float v) {
        buffer_.appendDouble(static_cast<double>(v));
        return *this;
    }

    LogItem& operator<<(double v) {
        buffer_.appendDouble(v);
        return *this;
    }

private:
    LogSeverity severity_;
    const char* file_;
    int line_;
    const char* function_;
    Logger* logger_;
    LogBuffer buffer_;
};

struct LoggerState {
    std::mutex mu_;
    LogSeverity min_severity_{LogSeverity::INFO};
    Logger* logger_{nullptr};
    std::unique_ptr<Logger> owned_logger_;
};

inline LoggerState& getLoggerState() {
    static LoggerState state;
    return state;
}

inline void setLogLevel(LogSeverity minSeverity) {
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);
    state.min_severity_ = minSeverity;
}

inline LogSeverity getLogLevel() {
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);
    return state.min_severity_;
}

inline void setGlobalLogger(Logger* logger) {
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);
    state.owned_logger_.reset();
    state.logger_ = logger;
}

inline Logger* getGlobalLogger() {
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);
    return state.logger_;
}

inline bool isLogEnabled(LogSeverity severity) {
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);
    if (state.logger_ == nullptr) return false;
    return static_cast<uint8_t>(severity) >= static_cast<uint8_t>(state.min_severity_);
}

inline void initializeLogger(LogChoice choice = LogChoice::STDERR,
                             const char* logFilePath = nullptr,
                             LogSeverity minSeverity = LogSeverity::INFO)
{
    auto& state = getLoggerState();
    std::lock_guard<std::mutex> lock(state.mu_);

    state.min_severity_ = minSeverity;

    switch (choice) {
        case LogChoice::NONE: {
            state.owned_logger_ = std::make_unique<NullLogger>();
            state.logger_ = state.owned_logger_.get();
            break;
        }
        case LogChoice::FILE: {
            state.owned_logger_ = std::make_unique<FileLogger>(logFilePath);
            state.logger_ = state.owned_logger_.get();
            break;
        }
        case LogChoice::STDOUT: {
            state.owned_logger_ = std::make_unique<StdStreamLogger>(stdout);
            state.logger_ = state.owned_logger_.get();
            break;
        }
        case LogChoice::STDERR: {
            state.owned_logger_ = std::make_unique<StdStreamLogger>(stderr);
            state.logger_ = state.owned_logger_.get();
            break;
        }
        case LogChoice::CUSTOM: {
            // User must call setGlobalLogger() separately.
            state.owned_logger_.reset();
            // Keep existing state.logger_ as-is.
            break;
        }
        default: {
            state.owned_logger_ = std::make_unique<StdStreamLogger>(stderr);
            state.logger_ = state.owned_logger_.get();
            break;
        }
    }
}

} // namespace astervec

// Logging macros: allow LOG(INFO) instead of LOG(astervec::LogSeverity::INFO).
#define LOG(severity) \
    if (!::astervec::isLogEnabled(::astervec::LogSeverity::severity)) ; \
    else ::astervec::LogItem(::astervec::LogSeverity::severity, __FILE__, __LINE__, __func__, ::astervec::getGlobalLogger())

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            LOG(ERR) << "Check failed: " << #condition; \
            throw std::runtime_error("Check failed: it's either a bug or inconsistent data!"); \
        } \
    } while (0)

#define CHECK_MSG(condition, message) \
    do { \
        if (!(condition)) { \
            LOG(ERR) << "Check failed: " << #condition << " " << (message); \
            throw std::runtime_error(std::string("Check failed: ") + (message)); \
        } \
    } while (0)

#ifndef NDEBUG
#define DLOG(severity) LOG(severity)
#define DCHECK(condition) CHECK(condition)
#else
#define DLOG(severity) while (0) LOG(severity)
#define DCHECK(condition) while (0) CHECK(condition)
#endif

/*
Usage example:

#include "Logger.h"

int main() {
    astervec::initializeLogger(astervec::LogChoice::STDERR, nullptr, astervec::LogSeverity::INFO);

    LOG(INFO) << "Hello";
    DLOG(DEBUG) << "Debug only in non-release build";

    CHECK(1 + 1 == 2);
    // LOG(FATAL) << "This will abort";

    return 0;
}
*/

#endif // LSM_VEC_LOGGER_H_
