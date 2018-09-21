#include <vector>
#include <cmath>
#include <iostream>


int main() {
    std::vector<double> x(11);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        x[i] = 0.1*i;
        y[i] = std::sin(x[i]);
    }

    double input_x = 0.34;
    // Find index such that x[i] < input_x < x[i + 1]
    // Assume x is sorted in increasing order

    // How to avoid O(n) lookup time?
    for (std::size_t i = 0; i < x.size() - 1; i++) {
        if (x[i] < input_x && input_x < x[i + 1]) {
            std::cout <<  y[i] + (input_x - x[i])*(y[i + 1] - y[i])/(x[i + 1] - x[i]) << std::endl;
        }
    }
}
