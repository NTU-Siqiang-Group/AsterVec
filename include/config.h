#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstdint>
#include <sstream>
#include <cstdlib>

struct Config {
    // --- Index / graph hyper-parameters ---
    int   M = 8;
    int   Mmax = 16;
    int   Ml = 1;
    float efConstruction = 64.0f;
    size_t input_size;
    int random_seed = 12345;
    bool enable_stats = false;
    int k = 1;
    bool reinit = true;

    int vector_storage_type = 0; // 0 for basic, 1 for paged

    // --- Paths / I/O ---
    std::string db_path;                // required: RocksDB path (directory)
    uint64_t    db_target_size = 107374182400ULL; // default: 100 GiB

    // If --vec is omitted, this defaults to db_path
    std::string vector_file_path;
    size_t vec_file_capacity = 100000;
    size_t paged_max_cached_pages = 512;

    // New: single data directory and optional shared name/prefix
    // If only --data-dir is provided (no --name), filenames become:
    //   <data-dir>/input.fvecs, <data-dir>/query.fvecs, <data-dir>/groundtruth.ivecs
    // If both --data-dir and --name are provided, filenames become:
    //   <data-dir>/<name>_input.fvecs, <data-dir>/<name>_query.fvecs, <data-dir>/<name>_groundtruth.ivecs
    std::string data_dir;  // directory containing dataset files
    std::string data_name; // optional prefix/base name

    // Explicit file paths (still supported; override any derived path)
    std::string input_file_path;        // base/input (.fvecs)
    std::string query_file_path;        // query (.fvecs)
    std::string groundtruth_file_path;  // groundtruth (.ivecs)

    std::string output_path = "output.txt";
    std::string edge_update_policy = "eager"; 

    // Defaults for suffixes and extensions (kept simple; can be made configurable later)
    std::string input_suffix = "input";
    std::string query_suffix = "query";
    std::string truth_suffix = "groundtruth";
    std::string input_ext = ".fvecs";
    std::string query_ext = ".fvecs";
    std::string truth_ext = ".ivecs";

    static Config ForDatabase(const std::string& db_path_value,
                              const std::string& vector_file_path_value,
                              size_t vec_capacity_value,
                              size_t paged_cache_pages_value,
                              int vector_storage_type_value,
                              uint64_t db_target_size_value,
                              int random_seed_value,
                              bool reinit_value = false) {
        Config cfg_;
        cfg_.db_path = db_path_value;
        cfg_.vector_file_path = vector_file_path_value;
        cfg_.vec_file_capacity = vec_capacity_value;
        cfg_.paged_max_cached_pages = paged_cache_pages_value;
        cfg_.vector_storage_type = vector_storage_type_value;
        cfg_.db_target_size = db_target_size_value;
        cfg_.random_seed = random_seed_value;
        cfg_.reinit = reinit_value;
        return cfg_;
    }

    static void PrintHelp(const char* prog) {
        std::cout <<
R"(Usage:
  )" << prog << R"( --db <path> [--vec <path>] \
       (--data-dir <dir> [--name <prefix>] | --base <fvecs> --query <fvecs> --truth <ivecs>) \
       [--M <int>] [--Mmax <int>] [--Ml <int>] [--efc <float>] \
       [--paged-cache-pages <count>] \
       [--db-target-size <bytes>] [--out <file>] \
       [--reinit <0|1>] \
       [--k <int>] \
       [--stats <0|1>] \
       [--edge-policy <eager|lazy|none>] \
       [-h|--help]

Notes:
  • If --vec is omitted, it defaults to the value of --db.
  • If you provide --data-dir without --name, the tool expects:
       <data-dir>/input.fvecs, <data-dir>/query.fvecs, <data-dir>/groundtruth.ivecs
  • If you provide both --data-dir and --name, the tool expects:
       <data-dir>/<name>_input.fvecs, <data-dir>/<name>_query.fvecs, <data-dir>/<name>_groundtruth.ivecs
  • You may still override any file via --base/--query/--truth explicitly.

Short aliases:
  -d --db, -v --vec, -D --data-dir, -n --name,
  -i --base, -q --query, -g --truth,
  -m --M, -x --Mmax, -l --Ml, -e --efc, -k --k,
  -s --db-target-size, -o --out, -p --edge-policy
)";
    }

    static bool isFlag(const std::string& a) { return a.size() > 1 && a[0] == '-'; }

    static Config Parse(int argc, char* argv[]) {
        Config cfg_;
        std::unordered_map<std::string, std::string> kv;

        auto put = [&](const std::string& k, const std::string& v){
            if (!v.empty()) {
                kv[k] = v;
                return;
            }
            if (k == "stats") {
                kv[k] = "1";
            }
            if (k == "reinit") {
                kv[k] = "1";
            }
        };

        // Parse GNU-style flags
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "-h" || arg == "--help") {
                PrintHelp(argv[0]);
                std::exit(0);
            }
            if (arg.rfind("--", 0) == 0) {
                auto eq = arg.find('=');
                if (eq != std::string::npos) {
                    put(arg.substr(2, eq - 2), arg.substr(eq + 1));
                } else {
                    std::string key = arg.substr(2);
                    std::string val;
                    if (i + 1 < argc && !isFlag(argv[i+1])) { val = argv[++i]; }
                    put(key, val);
                }
            } else if (arg.rfind("-", 0) == 0) {
                std::string key = arg.substr(1);
                std::string val;
                if (i + 1 < argc && !isFlag(argv[i+1])) { val = argv[++i]; }
                if (key == "d") put("db", val);
                else if (key == "s") put("db-target-size", val);
                else if (key == "v") put("vec", val);
                else if (key == "D") put("data-dir", val);
                else if (key == "n") put("name", val);
                else if (key == "i") put("base", val);
                else if (key == "q") put("query", val);
                else if (key == "g") put("truth", val);
                else if (key == "o") put("out", val);
                else if (key == "m") put("M", val);
                else if (key == "x") put("Mmax", val);
                else if (key == "l") put("Ml", val);
                else if (key == "e") put("efc", val);
                else if (key == "k") put("k", val);
                else if (key == "p") put("edge-policy", val);
                else if (key == "V") put("vec-storage", val);
            }
        }

        // Load scalar values
        if (kv.count("db"))                 cfg_.db_path = kv["db"];
        if (kv.count("db-target-size"))     cfg_.db_target_size = parseU64(kv["db-target-size"]);
        if (kv.count("vec"))                cfg_.vector_file_path = kv["vec"];
        if (kv.count("data-dir"))           cfg_.data_dir = kv["data-dir"];
        if (kv.count("name"))               cfg_.data_name = kv["name"];

        if (kv.count("out"))                cfg_.output_path = kv["out"];
        if (kv.count("edge-policy"))        cfg_.edge_update_policy = kv["edge-policy"];
        if (kv.count("stats"))              cfg_.enable_stats = (kv["stats"] != "0");
        if (kv.count("reinit"))             cfg_.reinit = (kv["reinit"] != "0");

        if (kv.count("M"))                  cfg_.M = parseI(kv["M"]);
        if (kv.count("Mmax"))               cfg_.Mmax = parseI(kv["Mmax"]);
        if (kv.count("Ml"))                 cfg_.Ml = parseI(kv["Ml"]);
        if (kv.count("efc"))                cfg_.efConstruction = parseF(kv["efc"]);
        if (kv.count("k"))                  cfg_.k = parseI(kv["k"]);
        if (kv.count("paged-cache-pages"))  cfg_.paged_max_cached_pages = parseU64(kv["paged-cache-pages"]);

        // Explicit file overrides (take precedence over any derived path)
        if (kv.count("base"))               cfg_.input_file_path = kv["base"];
        if (kv.count("query"))              cfg_.query_file_path = kv["query"];
        if (kv.count("truth"))              cfg_.groundtruth_file_path = kv["truth"];
        if (kv.count("vec-storage"))              cfg_.vector_storage_type = parseI(kv["vec-storage"]);

        // Default vector_file_path to db_path if not provided
        if (cfg_.vector_file_path.empty() && !cfg_.db_path.empty()) {
            cfg_.vector_file_path = cfg_.db_path + "/vector.log";
        }

        // Validate we have some way to locate data files:
        // Either (A) all three explicit paths are given, or (B) a data-dir is given
        // (with optional name) so we can derive the three paths.
        bool have_all_explicit =
            !cfg_.input_file_path.empty() &&
            !cfg_.query_file_path.empty() &&
            !cfg_.groundtruth_file_path.empty();

        if (!have_all_explicit) {
            if (cfg_.data_dir.empty()) {
            std::ostringstream oss;
            oss << "Missing dataset specification. Provide either:\n"
                << "  (A) --base --query --truth, or\n"
                << "  (B) --data-dir [--name]\n\n";
            std::cerr << oss.str();
            PrintHelp(argv[0]);
            std::exit(1);
        }
            // Derive paths from data_dir (+ optional data_name)
            namespace fs = std::filesystem;
            auto join = [](const fs::path& a, const std::string& b) {
                return (a.string() + b);
            };
            auto stem = cfg_.data_name; // may be empty
            auto make_file = [&](const std::string& suffix, const std::string& ext){
                if (stem.empty()) return suffix + ext;              // e.g., "input.fvecs"
                return stem + "_" + suffix + ext;                    // e.g., "<name>_input.fvecs"
            };

            if (cfg_.input_file_path.empty())
                cfg_.input_file_path = join(cfg_.data_dir, make_file(cfg_.input_suffix, cfg_.input_ext));
            if (cfg_.query_file_path.empty())
                cfg_.query_file_path = join(cfg_.data_dir, make_file(cfg_.query_suffix, cfg_.query_ext));
            if (cfg_.groundtruth_file_path.empty())
                cfg_.groundtruth_file_path = join(cfg_.data_dir, make_file(cfg_.truth_suffix, cfg_.truth_ext));
        }

        // Validate required root options
        std::vector<std::string> missing;
        if (cfg_.db_path.empty())               missing.push_back("--db");
        if (cfg_.vector_file_path.empty())      missing.push_back("--vec"); // normally filled from --db
        if (!missing.empty()) {
            std::ostringstream oss;
            oss << "Missing required options:";
            for (auto& m : missing) oss << " " << m;
            oss << "\n\n";
            std::cerr << oss.str();
            PrintHelp(argv[0]);
            std::exit(1);
        }

        // Filesystem prep/validation
        namespace fs = std::filesystem;

        // Ensure DB directory exists
        try {
            fs::create_directories(fs::path(cfg_.db_path));
        } catch (...) {
            std::cerr << "Failed to create db path: " << cfg_.db_path << "\n";
            std::exit(1);
        }

        // Ensure parent directory for vector path exists
        if (!cfg_.vector_file_path.empty()) {
            auto parent = fs::path(cfg_.vector_file_path).parent_path();
            if (!parent.empty()) {
                try { fs::create_directories(parent); } catch (...) {
                    std::cerr << "Failed to create parent dir for vec: " << parent << "\n";
                    std::exit(1);
                }
            }
        }

        // Required input files must exist
        auto must_exist = [&](const std::string& p, const char* what){
            if (!fs::exists(p)) {
                std::cerr << "File not found for " << what << ": " << p << "\n";
                std::exit(1);
            }
        };
        must_exist(cfg_.input_file_path, "input(.fvecs)");
        must_exist(cfg_.query_file_path, "query(.fvecs)");
        must_exist(cfg_.groundtruth_file_path, "groundtruth(.ivecs)");

        // Ensure parent directory for output file exists
        if (!cfg_.output_path.empty()) {
            auto parent = fs::path(cfg_.output_path).parent_path();
            if (!parent.empty()) {
                try { fs::create_directories(parent); } catch (...) {
                    std::cerr << "Failed to create parent dir for out: " << parent << "\n";
                    std::exit(1);
                }
            }
        }

        return cfg_;
    }

private:
    // Parsing helpers with error messages on bad input
    static int parseI(const std::string& s) {
        try { return std::stoi(s); } catch (...) { die("int", s); return 0;}
    }
    static uint64_t parseU64(const std::string& s) {
        try { return static_cast<uint64_t>(std::stoull(s)); } catch (...) { die("uint64", s); return 0ULL;}
    }
    static float parseF(const std::string& s) {
        try { return std::stof(s); } catch (...) { die("float", s); return 0.0f;}
    }
    static void die(const char* need, const std::string& got) {
        std::cerr << "Invalid " << need << " value: '" << got << "'\n";
        std::exit(1);
    }
};
