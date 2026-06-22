// Entrypoint for the AsterVec HTTP server.
//
// Configuration via env vars only (see include/astervec_http.h). No
// CLI flags — the deployment is "set env, docker run, done".

#include "astervec_http.h"

#include <iostream>
#include <string>

int main(int /*argc*/, char** /*argv*/) {
    astervec::HttpServerConfig cfg;
    std::string err;
    if (!astervec::LoadHttpConfigFromEnv(&cfg, &err)) {
        std::cerr << "{\"event\":\"config_error\",\"error\":\""
                  << err << "\"}\n";
        return 2;
    }
    return astervec::RunHttpServer(cfg);
}
