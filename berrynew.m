function main()
    % Define kx and ky ranges for calculation
    kx_vals = linspace(-pi, pi, 50); % Reduce grid density
    ky_vals = linspace(-pi, pi, 50);

    % Define parameters v, w, and t
    v = 1.0;
    w = 1.0;
    t = 1.0;

    % Compute Berry connection components
    [A_nx, A_ny] = berryConnection(kx_vals, ky_vals, v, w, t);

    % Compute Berry curvature Omega_n(k) using the curl of A_n(k)
    Omega_n = curl(A_nx, A_ny, kx_vals, ky_vals);

    % Smooth the Omega_n data using a simple averaging filter
    filter_size = 3;
    averaging_filter = ones(filter_size) / filter_size^2;
    Omega_n_smoothed = conv2(Omega_n, averaging_filter, 'same');

    % Calculate Berry phase for each grid point
    berry_phase = sum(sum(Omega_n_smoothed)) * (kx_vals(2) - kx_vals(1)) * (ky_vals(2) - ky_vals(1));

    % Display Berry phase value
    disp(['Berry Phase: ', num2str(berry_phase)]);

    % Plot contour plot of smoothed Berry curvature
    figure;
    subplot(1, 2, 1);
    contourf(kx_vals, ky_vals, Omega_n_smoothed, 20, 'LineColor', 'none');
    colormap('jet');
    colorbar;
    xlabel('kx');
    ylabel('ky');
    title('Contour Plot of Smoothed Berry Curvature \Omega_n(k)');

    % Plot Berry phase as a function of kx and ky
    [KX, KY] = meshgrid(kx_vals, ky_vals);
    subplot(1, 2, 2);
    surf(KX, KY, Omega_n_smoothed);
    xlabel('kx');
    ylabel('ky');
    zlabel('Berry Phase');
    title('Berry Phase as a Function of kx and ky');





    % Plot Berry connection components and Berry curvature
    figure;

    subplot(2, 2, 1);
    surf(kx_vals, ky_vals, abs(A_nx'));
    colorbar;
    xlabel('kx');
    ylabel('ky');
    zlabel('A_nx');
    title('Berry Connection A_nx(k)');

    subplot(2, 2, 2);
    surf(kx_vals, ky_vals, abs(A_ny'));
    colorbar;
    xlabel('kx');
    ylabel('ky');
    zlabel('A_ny');
    title('Berry Connection A_ny(k)');

    subplot(2, 2, 3);
    surf(kx_vals, ky_vals, abs(Omega_n'));
    colorbar;
    xlabel('kx');
    ylabel('ky');
    zlabel('Omega_n');
    title('Berry Curvature Omega_n(k)');

    subplot(2, 2, 4);
    surf(kx_vals, ky_vals, abs(angle(exp(1i * Omega_n)))');
    colorbar;
    xlabel('kx');
    ylabel('ky');
    zlabel('Phase');
    title('Berry Phase');
end

function [A_nx, A_ny] = berryConnection(kx_vals, ky_vals, v, w, t)
    A_nx = zeros(length(kx_vals), length(ky_vals));
    A_ny = zeros(length(kx_vals), length(ky_vals));

    for i = 1:length(kx_vals)
        for j = 1:length(ky_vals)
            % Compute eigenstates
            eigenstates = computeEigenstates(kx_vals(i), ky_vals(j), v, w, t);

            % Normalize eigenstates
            normalization_constant = sqrt(sum(abs(eigenstates).^2));
            psi = eigenstates / normalization_constant;

            % Compute gradient of phase using numerical differentiation
            grad = numericalGradient(@(kx, ky) angle(computeEigenstates(kx, ky, v, w, t)), kx_vals(i), ky_vals(j));
            A_nx(i, j) = grad(1);
            A_ny(i, j) = grad(2);
        end
    end
end

function eigenstates = computeEigenstates(kx, ky, v, w, t)
    H_matrix = [2 * t * cos(ky), v + w * exp(-1i * kx);
                v + w * exp(1i * kx), 2 * t * cos(ky)];
    [eigenvecs, ~] = eig(H_matrix);
    eigenstates = eigenvecs; % Get all eigenstates
end

function grad = numericalGradient(func, x, y)
    h = 1e-5;
    grad_x = (func(x + 1i * h, y) - func(x - 1i * h, y)) / (2 * h);
    grad_y = (func(x, y + 1i * h) - func(x, y - 1i * h)) / (2 * h);
    grad = [real(grad_x), real(grad_y)];
end

function Omega_n = curl(A_nx, A_ny, kx_vals, ky_vals)
    [dkx, dky] = gradient(A_ny, kx_vals, ky_vals);
    [dky, dkx] = gradient(A_nx, ky_vals, kx_vals);
    Omega_n = dkx - dky;
end
