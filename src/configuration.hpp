#pragma once

#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

#include "merian-nodes/graph/graph.hpp"

static const char* CONFIG_NAME = "merian-quake.json";
static const char* FALLBACK_CONFIG_NAME = "default_config.json";

static const char* CONFIG_PATH_ENV_VAR = "MERIAN_QUAKE_CONFIG_PATH";

class ConfigurationManager {
  public:
    ConfigurationManager(merian_nodes::Graph<>& graph, merian::FileLoader& loader)
        : graph(graph), loader(loader) {}

    void load() {
        std::string config_path =
            std::getenv(CONFIG_PATH_ENV_VAR) ? std::getenv(CONFIG_PATH_ENV_VAR) : CONFIG_NAME;
        if (std::filesystem::exists(config_path)) {
            SPDLOG_INFO("loading config {}", config_path);
        } else {
            auto default_config = loader.find_file(FALLBACK_CONFIG_NAME);
            assert(default_config.has_value());
            config_path = default_config.value().string();
            SPDLOG_DEBUG("loading default config {}", FALLBACK_CONFIG_NAME);
        }

        auto load = merian::JSONLoadProperties(std::filesystem::path(config_path));
        graph.properties(load);
    }
    void store() {
        auto dump = merian::JSONDumpProperties(CONFIG_NAME);
        graph.properties(dump);
    }
    void get(merian::Properties& config) {
        graph.properties(config);
    }

  private:
    merian_nodes::Graph<>& graph;
    merian::FileLoader& loader;
};
