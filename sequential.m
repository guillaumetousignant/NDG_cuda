clc;
clearvars;
close all

N = 36;
c = 1;
N_points = 1000;

[nodes, weights] = LegendreGaussNodesAndWeights(N); % OK
barycentric_weights = BarycentricWeights(nodes); % OK
lagrange_interpolating_polynomial_left = LagrangeInterpolatingPolynomials(-1, nodes, barycentric_weights); % OK
lagrange_interpolating_polynomial_right = LagrangeInterpolatingPolynomials(1, nodes, barycentric_weights); % OK
D = PolynomialDerivativeMatrix(nodes); % OK
D_hat = zeros(size(D));
for j = 0:N
    for i = 0:N
        D_hat(i+1, j+1) = -D(j+1, i+1) * weights(j+1)/weights(i+1);
    end
end
ksi = InterpolationPoints(N_points);
interpolation_matrix = PolynomialInterpolationMatrix(ksi, nodes, barycentric_weights);

delta_t = 0.00015;
save_times = [0.05, 0.1, 0.15];
t = 0;
t_end = save_times(end);
phi = zeros(N+1, 1);
for j = 0:N
    phi(j+1) = g(0, nodes(j+1), c);
end

figure('Name', 'Solution at different times', 'NumberTitle', 'off');
hold on
phi_interpolated = InterpolateToNewPoints(interpolation_matrix, phi);
plot(ksi, phi_interpolated, 'LineWidth', 2);
legends = {'t = 0'};

while t < (t_end + delta_t)
    phi = DGStepByRK3(t, delta_t, c, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right);
    t = t + delta_t;

    if (any((t >= save_times) & (t < save_times + delta_t)))
        phi_interpolated = InterpolateToNewPoints(interpolation_matrix, phi);
        plot(ksi, phi_interpolated, 'LineWidth', 2);
        legends = [legends, sprintf('t = %0.5g', t)];
    end
end
legend(legends, 'Location', 'best');
xlabel('x');
ylabel('u');
grid ON
title('Solution at different times');

%% Verification
y_test = -sin(pi * nodes);
u_L = InterpolateToBoundary(y_test, lagrange_interpolating_polynomial_left);
u_R = InterpolateToBoundary(y_test, lagrange_interpolating_polynomial_right);
y_prime = MxVDerivative(D, y_test);
y_prime_expected = -pi * cos(pi * nodes);

function y = g(t, x, c)
    sigma = 0.2;
    y = exp(-log(2) * (x - t * c)^2 /(sigma^2));
end


function [L_N, L_N_prime] = LegendrePolynomialAndDerivative(N, x)
    if N == 0
        L_N = 1;
        L_N_prime = 0;
    elseif N == 1
        L_N = x;
        L_N_prime = 1;
    else
        L_N_2 = 1;
        L_N_1 = x;
        L_N_2_prime = 0;
        L_N_1_prime = 1;

        for k = 2:N
            L_N = (2 * k - 1) * x * L_N_1/k - (k - 1) * L_N_2/k; % L_N_1(x) ??
            L_N_prime = L_N_2_prime + (2 * k - 1) * L_N_1;
            L_N_2 = L_N_1;
            L_N_1 = L_N;
            L_N_2_prime = L_N_1_prime;
            L_N_1_prime = L_N_prime;
        end
    end
end

function [nodes, weights] = LegendreGaussNodesAndWeights(N)
    nodes = zeros(N + 1, 1);
    weights = zeros(N + 1, 1);

    if N == 0
        nodes(1) = 0;
        weights(1) = 2;
    elseif N == 1
        nodes(1) = -sqrt(1/3);
        weights(1) = 1;
        nodes(2) = -nodes(1);
        weights(2) = weights(1);
    else
        for j = 0:((N + 1)/2 - 1)
            nodes(j + 1) = -cos(pi * (2 * j + 1)/(2 * N + 2));

            for k = 0:1000
                [L_N_plus1, L_N_plus1_prime] = LegendrePolynomialAndDerivative(N + 1, nodes(j + 1));
                delta = -L_N_plus1/L_N_plus1_prime;
                nodes(j + 1) = nodes(j + 1) + delta;
                if abs(delta) <= 0.00000001 * abs(nodes(j + 1))
                    break
                end
            end

            [~, L_N_plus1_prime] = LegendrePolynomialAndDerivative(N + 1, nodes(j + 1));
            nodes(N - j + 1) = -nodes(j + 1);
            weights(j + 1) = 2/((1 - nodes(j + 1)^2) * L_N_plus1_prime^2);
            weights(N - j + 1) = weights(j + 1);
        end
    end

    if mod(N, 2) == 0
        [~, L_N_plus1_prime] = LegendrePolynomialAndDerivative(N + 1, 0);
        nodes(N/2 + 1) = 0;
        weights(N/2 + 1) = 2/L_N_plus1_prime^2;
    end
end

function [nodes, weights] = ChebyshevGaussNodesAndWeights(N)
    nodes = zeros(N + 1, 1);
    weights = zeros(N + 1, 1);

    for j = 0:N
        nodes(j+1) = -cos(pi * (2 * j + 1)/(2 * N + 2));
        weights(j+1) = pi/(N + 1);
    end
end

function barycentric_weights = BarycentricWeights(nodes)
    barycentric_weights = zeros(length(nodes), 1);
    N = length(nodes) - 1;
    for j = 0:N
        barycentric_weights(j + 1) = 1;
    end

    for j = 1:N
        for k = 0:j-1
            barycentric_weights(k + 1) = barycentric_weights(k + 1) * (nodes(k + 1) - nodes(j + 1));
            barycentric_weights(j + 1) = barycentric_weights(j + 1) * (nodes(j + 1) - nodes(k + 1));
        end
    end

    for j = 0:N
        barycentric_weights(j + 1) = 1/barycentric_weights(j + 1);
    end
end

function lagrange_interpolating_polynomial = LagrangeInterpolatingPolynomials(x, nodes, barycentric_weights)
    lagrange_interpolating_polynomial = zeros(length(nodes), 1);
    N = length(nodes) - 1;

    xMatchNode = false;
    for j = 0:N
        lagrange_interpolating_polynomial(j + 1) = 0;
        if abs(x - nodes(j + 1))<=0.0000001
            lagrange_interpolating_polynomial(j + 1) = 1;
            xMatchNode = true;
        end
    end

    if xMatchNode
        return
    end

    s = 0;
    for j = 0:N
        t = barycentric_weights(j + 1)/(x - nodes(j + 1));
        lagrange_interpolating_polynomial(j + 1) = t;
        s = s + t;
    end

    for j = 0:N
        lagrange_interpolating_polynomial(j + 1) = lagrange_interpolating_polynomial(j + 1)/s;
    end
end

function D = PolynomialDerivativeMatrix(nodes)
    D = zeros(length(nodes), length(nodes));
    barycentric_weights = BarycentricWeights(nodes);
    N = length(nodes) - 1;

    for i = 0:N
        D(i+1, i+1) = 0;
        for j = 0:N
            if i ~= j
                D(i+1, j+1) = barycentric_weights(j + 1)/(barycentric_weights(i + 1) * (nodes(i+1) - nodes(j+1)));
                D(i+1, i+1) = D(i+1, i+1) - D(i+1, j+1);
            end
        end
    end
end

function derivative = MxVDerivative(D, f) % OK
    derivative = zeros(length(f), 1);
    N = length(f) - 1;

    for i = 0:N
        t = 0;
        for j = 0:N
            t = t + D(i+1, j+1) * f(j+1);
        end
        derivative(i+1) = t;
    end
end

function phi_prime = ComputeDGDerivative(phi_L, phi_R, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right)
    N = length(phi) - 1;
    phi_prime = MxVDerivative(D_hat, phi);

    for j = 0:N
        phi_prime(j+1) = phi_prime(j+1) + (phi_R * lagrange_interpolating_polynomial_right(j+1) - phi_L * lagrange_interpolating_polynomial_left(j+1))/weights(j+1);
    end
end

function interpolated_value = InterpolateToBoundary(phi, lagrange_interpolating_polynomial)
    interpolated_value = 0;
    N = length(phi) - 1;
    for j = 0:N
        interpolated_value = interpolated_value + lagrange_interpolating_polynomial(j+1) * phi(j+1);
    end
end

function phi_dot = DGTimeDerivative(t, c, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right)
    if c > 0
        phi_L = g(t, -1, c);
        phi_R = InterpolateToBoundary(phi, lagrange_interpolating_polynomial_right);
    else
        phi_L = InterpolateToBoundary(phi, lagrange_interpolating_polynomial_left);
        phi_R =  g(t, 1, c);
    end

    phi_dot = -c * ComputeDGDerivative(phi_L, phi_R, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right);
end

function phi = DGStepByRK3(t_start, delta_t, c, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right)
    N = length(phi) - 1;
    G = zeros(length(phi), 1);
    a = [0, -5/9, -153/128];
    b = [0, 1/3, 3/4];
    g_rk = [1/3, 15/16, 8/15];

    for m = 1:3
        t = t_start + b(m) * delta_t;
        phi_dot = DGTimeDerivative(t, c, phi, D_hat, weights, lagrange_interpolating_polynomial_left, lagrange_interpolating_polynomial_right);
        for j = 0:N
            G(j+1) = a(m) * G(j+1) + phi_dot(j+1);
            phi(j+1) = phi(j+1) + g_rk(m) * delta_t * G(j+1);
        end
    end
end

function ksi = InterpolationPoints(N_interpolation_points)
    ksi = zeros(N_interpolation_points, 1);
    for k = 0:N_interpolation_points-1
        ksi(k+1) = 2 * k / (N_interpolation_points - 1) - 1;
    end
end

% Algorithm 32
function interpolation_matrix = PolynomialInterpolationMatrix(ksi, nodes, barycentric_weights) 
    N = length(nodes) - 1;
    N_interpolation_points = length(ksi);
    interpolation_matrix = zeros(length(ksi), length(nodes));

    for k = 0:N_interpolation_points-1
        row_has_match = false;
        for j = 0:N
            interpolation_matrix(k+1, j+1) = 0;
            if abs(ksi(k+1) - nodes(j + 1))<=0.0000001
                row_has_match = true;
                interpolation_matrix(k+1, j+1) = 1;
            end
        end

        if ~row_has_match
            s = 0;
            for j = 0:N
                t = barycentric_weights(j+1)/(ksi(k+1) - nodes(j+1));
                interpolation_matrix(k+1, j+1) = t;
                s = s + t;
            end
            for j = 0:N
                interpolation_matrix(k+1, j+1) = interpolation_matrix(k+1, j+1)/s;
            end
        end
    end
end

% Algorithm 33
function phi_interpolated = InterpolateToNewPoints(interpolation_matrix, phi)
    N_interpolation_points = size(interpolation_matrix, 1);
    N = length(phi) - 1;
    phi_interpolated = zeros(N_interpolation_points);
    for i = 0:N_interpolation_points-1
        t = 0;
        for j = 0:N
            t = t + interpolation_matrix(i+1, j+1) * phi(j+1);
        end
        phi_interpolated(i+1) = t;
    end
end