#include <exception>
#include <iostream>

#include <boost/locale.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"

namespace po = boost::program_options;

po::options_description po_create_desc() {
    po::options_description desc("all the options");
    desc.add_options()
        ("help", "print help message")
    ;

    return desc;
}

int po_fill_vm(const po::options_description& desc, po::variables_map& vm, const int argc, char* const* argv, const std::string& program_name) {
    try {
        po::store(
            po::command_line_parser(argc, argv).options(desc).run(),
            vm
        );
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    if (vm.count("help")) {
        std::cout << program_name << std::endl
            << std::endl
            << desc << std::endl;
        return 1;
    }

    try {
        po::notify(vm);
    } catch(std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int gen_locale(std::locale& loc) {
    boost::locale::generator gen;
    loc = gen("");

    // before we start, check if we're working on an UTF8 system
    if (!std::use_facet<boost::locale::info>(loc).utf8()) {
        std::cerr << "sorry, this program only works on UTF8 systems" << std::endl;
        return 1;
    }

    return 0;
}
