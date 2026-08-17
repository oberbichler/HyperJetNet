using System;

namespace HyperJet.Tests
{
    /// <summary>
    /// High-accuracy finite-difference reference derivatives, used to validate the analytic
    /// derivative formulas of the AD kernels against an independent numerical ground truth.
    /// </summary>
    /// <remarks>
    /// All stencils are 4th-order accurate. With <see cref="H"/> = 1e-3 and well-scaled functions the
    /// residual error is roughly 1e-12 (gradient) and 1e-9 (Hessian), which is far below the
    /// tolerances used in the tests.
    /// </remarks>
    public static class NumericDiff
    {
        public const double H = 1e-3;

        /// <summary>4th-order central difference for df/dx.</summary>
        public static double D1(Func<double, double, double> f, double x, double y, int axis)
        {
            double F(double t) => axis == 0 ? f(x + t, y) : f(x, y + t);
            return (8.0 * (F(H) - F(-H)) - (F(2 * H) - F(-2 * H))) / (12.0 * H);
        }

        /// <summary>4th-order central difference for d²f/dx².</summary>
        public static double D2(Func<double, double, double> f, double x, double y, int axis)
        {
            double F(double t) => axis == 0 ? f(x + t, y) : f(x, y + t);
            return (-F(2 * H) + 16.0 * F(H) - 30.0 * F(0.0) + 16.0 * F(-H) - F(-2 * H)) / (12.0 * H * H);
        }

        /// <summary>
        /// Mixed partial d²f/dxdy. The plain 4-point stencil is only 2nd-order accurate, so two
        /// step sizes are combined by Richardson extrapolation to reach 4th order.
        /// </summary>
        public static double D2Mixed(Func<double, double, double> f, double x, double y)
        {
            double Cross(double h) =>
                (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4.0 * h * h);

            double d = Cross(H);
            double d2 = Cross(2.0 * H);
            return (4.0 * d - d2) / 3.0;
        }
    }
}
