#ifndef MESH_T_H
#define MESH_T_H

class Mesh_t {
public:
    Mesh_t(int N_elements, int initial_N, float x_min, float x_max) : N_elements_(N_elements), N_faces_(N_elements), initial_N_(initial_N);
    ~Mesh_t();

    int N_elements_;
    int N_faces_;
    int initial_N_;
    Element_t* elements_;
    Face_t* faces_;

    void set_initial_conditions(const float* nodes);
    void print();
    void write_file_data(int N_points, float time, const float* velocity, const float* coordinates);
    void write_data(float time, int N_interpolation_points, const float* interpolation_matrices);
    void solve(const float delta_t, const std::vector<float> output_times, const NDG_t &NDG);
};

#endif