/* Fast Euler-Maruyama loop for polynomial SDE simulation. */
#include <math.h>

int em_loop(double *omega, double *theta, const double *dW,
            double c0, double c1, double c2, double c3, double c4,
            double c5, double c6, double c7, double c8, double c9,
            double sigma, double dt, int n_steps) {
    int n_clamp = 0;
    for (int i = 1; i < n_steps; i++) {
        double th = theta[i - 1];
        double om = omega[i - 1];

        double f = c0 + th * (c1 + th * (c3 + th * c6))
                 + om * (c2 + th * (c4 + th * c7) + om * (c5 + th * c8 + om * c9));

        theta[i] = th + om * dt;
        omega[i] = om + f * dt + sigma * dW[i];

        if (omega[i] > 2.0) { omega[i] = 2.0; n_clamp++; }
        else if (omega[i] < -2.0) { omega[i] = -2.0; n_clamp++; }
        if (theta[i] > 100.0) { theta[i] = 100.0; n_clamp++; }
        else if (theta[i] < -100.0) { theta[i] = -100.0; n_clamp++; }
    }
    return n_clamp;
}
