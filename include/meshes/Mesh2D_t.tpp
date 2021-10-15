template<typename Polynomial>
auto SEM::Host::Meshes::Mesh2D_t::estimate_error(const std::vector<std::vector<hostFloat>>& polynomial_nodes, const std::vector<std::vector<hostFloat>>& weights) -> void {
    for (size_t element_index = 0; element_index < n_elements_; ++element_index) {
        elements_[element_index].estimate_error<Polynomial>(tolerance_min_, tolerance_max_, polynomial_nodes[elements_[element_index].N_], weights[elements_[element_index].N_]);
    }
}