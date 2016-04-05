#include <iostream>
#include <memory>
#include <vector>

#include "wavelet.hpp"

int main() {
    int n = 4;
    int j = 2;
    auto w = std::make_shared<wavelet>("haar");
    w->print_summary();

    wavelet_transform mywt(w, "dwt", n, j);
    mywt.extension("per");
    mywt.conv("direct");

    std::vector<double> data{2, 0, 1, 7};
    mywt.run_dwt(data.data());
    mywt.print_summary();

    for (std::size_t i = 0; i < mywt.outlength(); ++i) {
        std::cout << mywt.output()[i] << std::endl;
    }
}
