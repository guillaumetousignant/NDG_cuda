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
            auto getCmdOptionOr(const std::string &option, const T &default) const -> T {
                const std::string result = getCmdOption(option);
                if (result.empty()) {
                    return default;
                }
                else {
                    std::stringstream sstream(result);
                    T result;
                    sstream >> result;
                    return result;
                }
            }

        private:
            std::vector <std::string> tokens_;
    };
}}

#endif
