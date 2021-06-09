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
            auto getCmdOptionOr(const std::string &option, T default) const -> T {
                std::stringstream sstream(getCmdOption(option));
                if (sstream.eof()) {
                    return default;
                }
                else {
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
