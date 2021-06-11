#ifndef NDG_INPUTPARSER_T_H
#define NDG_INPUTPARSER_T_H

#include <vector>
#include <string>
#include <sstream>

namespace SEM { namespace Helpers {
    class InputParser_t {
        public:
            InputParser_t(int &argc, char **argv);

            auto getCmdOption(const std::string &option) const -> std::string;

            auto cmdOptionExists(const std::string &option) const -> bool;

            template<typename T>
            auto getCmdOptionOr(const std::string &option, const T &default_value) const -> T {
                auto itr = std::find(tokens_.begin(), tokens_.end(), option);
                if (itr != tokens_.end() && ++itr != tokens_.end()){
                    std::stringstream sstream(*itr);
                    T result_value;
                    sstream >> result_value;
                    return result_value;
                }
                return default_value;
            }

        private:
            std::vector <std::string> tokens_;
    };
}}

#endif
