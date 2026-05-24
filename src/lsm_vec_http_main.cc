// Entrypoint for the LSM-Vec HTTP server.
//
// Configuration via env vars only (see include/lsm_vec_http.h). No
// CLI flags — the deployment is "set env, docker run, done".

#include "lsm_vec_http.h"

#include <iostream>
#include <string>

int main(int /*argc*/, char** /*argv*/) {
    lsm_vec::HttpServerConfig cfg;
    std::string err;
    if (!lsm_vec::LoadHttpConfigFromEnv(&cfg, &err)) {
        std::cerr << "{\"event\":\"config_error\",\"error\":\""
                  << err << "\"}\n";
        return 2;
    }
    return lsm_vec::RunHttpServer(cfg);
}
