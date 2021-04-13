#include "helpers/InputParser_t.h"

SEM::Helpers::InputParser_t::InputParser_t (int &argc, char **argv) {
    for (int i=1; i < argc; ++i) {
        tokens_.push_back(std::string(argv[i]));
    }
}

/// @author iain
std::string SEM::Helpers::InputParser_t::getCmdOption(const std::string &option) const {
    auto itr = std::find(tokens_.begin(), tokens_.end(), option);
    if (itr != tokens_.end() && ++itr != tokens_.end()){
        return *itr;
    }
    return "";
}

/// @author iain
bool SEM::Helpers::InputParser_t::cmdOptionExists(const std::string &option) const {
    return std::find(tokens_.begin(), tokens_.end(), option) != tokens_.end();
}