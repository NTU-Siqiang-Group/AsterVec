#pragma once

// Small POSIX helpers for making files and directory entries durable.

#include <fcntl.h>
#include <unistd.h>

#include <string>

namespace astervec {
namespace durable {

// fsync the file at `path`. fsync on any descriptor flushes the file's data,
// so this also covers writes made through other descriptors or streams
// (after those streams have been flushed).
inline bool fsync_file(const std::string& path)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

// fsync a directory so that renames and new entries inside it survive a
// power loss.
inline bool fsync_dir(const std::string& dir)
{
    int fd = ::open(dir.c_str(), O_RDONLY);
    if (fd < 0) return false;
    bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

inline bool file_exists(const std::string& path)
{
    return ::access(path.c_str(), F_OK) == 0;
}

}  // namespace durable
}  // namespace astervec
