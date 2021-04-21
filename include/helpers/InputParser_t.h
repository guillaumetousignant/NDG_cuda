#ifndef NDG_INPUTPARSER_T_H
#define NDG_INPUTPARSER_T_H

#include <vector>
#include <string>

namespace SEM { namespace Helpers {
    class InputParser_t {
        public:
            InputParser_t(int &argc, char **argv);

            auto getCmdOption(const std::string &option) const -> std::string;

            auto cmdOptionExists(const std::string &option) const -> bool;

        private:
            std::vector <std::string> tokens_;
    };
}}

#endif
