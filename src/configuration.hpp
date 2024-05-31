#pragma once

#include "merian/utils/configuration_json_dump.hpp"
#include "merian/utils/configuration_json_load.hpp"

#include "processing_graph.hpp"

static const char* CONFIG_NAME = "merian-quake.json";
static const char* FALLBACK_CONFIG_NAME = "default_config.json";

static const char* CONFIG_PATH_ENV_VAR = "MERIAN_QUAKE_CONFIG_PATH";

class ConfigurationManager {
  public:
    ConfigurationManager(ProcessingGraph& graph, merian::FileLoader& loader)
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

        auto load = merian::JSONLoadConfiguration(std::filesystem::path(config_path));
        graph.get().configuration(load);
    }
    void store() {
        auto dump = merian::JSONDumpConfiguration(CONFIG_NAME);
        graph.get().configuration(dump);
    }
    void get(merian::Configuration& config) {
        graph.get().configuration(config);
    }

  private:
    ProcessingGraph& graph;
    merian::FileLoader& loader;
};
