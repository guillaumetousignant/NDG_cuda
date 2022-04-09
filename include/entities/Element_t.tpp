#include <cmath>

template<typename Polynomial>
void SEM::Host::Entities::Element_t::estimate_error(const std::vector<std::vector<hostFloat>>& nodes, const std::vector<std::vector<hostFloat>>& weights) {
    for (int k = 0; k <= N_; ++k) {
        intermediate_[k] = 0.0;
        for (int i = 0; i <= N_; ++i) {
            const hostFloat L_N = Polynomial::polynomial(k, nodes[N_][i]);

            intermediate_[k] += (2 * k + 1) * 0.5 * phi_[i] * L_N * weights[N_][i];
        }
        intermediate_[k] = std::abs(intermediate_[k]);
    }

    constexpr hostFloat tolerance_min = 1e25;     // Refine above this
    constexpr hostFloat tolerance_max = 1e-25;    // Coarsen below this

    const hostFloat C = exponential_decay();

    // sum of error
    error_ = std::sqrt(std::pow(C, 2) * 0.5/sigma_) * std::exp(-sigma_ * (N_ + 1));

    if(error_ > tolerance_min) {    // need refine
        refine_ = true;
        coarsen_ = false;
    }
    else if(error_ <= tolerance_max ) { // need coarsen
        refine_ = false;
        coarsen_ = true;
    }
    else {  // if error in between then do nothing
        refine_ = false;
        coarsen_ = false;
    }
}
