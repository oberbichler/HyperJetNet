using System;
using System.Collections.Generic;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;

namespace HyperJet.Tests
{
    /// <summary>
    /// Validates every generated math function of <c>DDScalar2&lt;double&gt;</c> against finite
    /// differences of the corresponding plain-<c>double</c> function.
    /// </summary>
    /// <remarks>
    /// Each function is applied to a non-trivial inner expression
    /// <c>a(x, y) = s * (1.1x + 0.9y + 0.3xy)</c> rather than to a bare variable. That exercises the
    /// full chain rule — a wrong second-derivative coefficient stays invisible when the inner
    /// gradient is a unit vector and the inner Hessian is zero.
    /// </remarks>
    public class MathFunctionDerivativeTests
    {
        private const double X0 = 0.5;
        private const double Y0 = 0.4;

        /// <summary>Inner expression, unscaled. <c>Raw(X0, Y0) == 0.97</c>.</summary>
        private static double RawA(double x, double y) => 1.1 * x + 0.9 * y + 0.3 * x * y;
        private static DD2 RawA(DD2 x, DD2 y) => 1.1 * x + 0.9 * y + 0.3 * x * y;

        /// <summary>Second, independent inner expression for binary functions. <c>RawB(X0, Y0) == 0.71</c>.</summary>
        private static double RawB(double x, double y) => 0.8 * x - 0.6 * y + 0.25 * x * y + 0.5;
        private static DD2 RawB(DD2 x, DD2 y) => 0.8 * x - 0.6 * y + 0.25 * x * y + 0.5;

        private const double RawA0 = 1.1 * X0 + 0.9 * Y0 + 0.3 * X0 * Y0;
        private const double RawB0 = 0.8 * X0 - 0.6 * Y0 + 0.25 * X0 * Y0 + 0.5;

        private sealed record UnaryCase(Func<DD2, DD2> Ad, Func<double, double> Scalar, double Target);
        private sealed record BinaryCase(Func<DD2, DD2, DD2> Ad, Func<double, double, double> Scalar, double TargetA, double TargetB);

        private static readonly Dictionary<string, UnaryCase> UnaryCases = new()
        {
            ["Sin"] = new(DD2.Sin, Math.Sin, 0.7),
            ["Cos"] = new(DD2.Cos, Math.Cos, 0.7),
            ["Tan"] = new(DD2.Tan, Math.Tan, 0.7),
            ["Asin"] = new(DD2.Asin, Math.Asin, 0.37),
            ["Acos"] = new(DD2.Acos, Math.Acos, 0.37),
            ["Atan"] = new(DD2.Atan, Math.Atan, 0.7),
            ["SinPi"] = new(DD2.SinPi, t => Math.Sin(Math.PI * t), 0.37),
            ["CosPi"] = new(DD2.CosPi, t => Math.Cos(Math.PI * t), 0.37),
            ["TanPi"] = new(DD2.TanPi, t => Math.Tan(Math.PI * t), 0.37),
            ["AsinPi"] = new(DD2.AsinPi, t => Math.Asin(t) / Math.PI, 0.37),
            ["AcosPi"] = new(DD2.AcosPi, t => Math.Acos(t) / Math.PI, 0.37),
            ["AtanPi"] = new(DD2.AtanPi, t => Math.Atan(t) / Math.PI, 0.7),
            ["Exp"] = new(DD2.Exp, Math.Exp, 0.7),
            ["Exp2"] = new(DD2.Exp2, t => Math.Pow(2.0, t), 0.7),
            ["Exp10"] = new(DD2.Exp10, t => Math.Pow(10.0, t), 0.7),
            ["ExpM1"] = new(DD2.ExpM1, t => Math.Exp(t) - 1.0, 0.7),
            ["Log"] = new(DD2.Log, Math.Log, 1.7),
            ["Log2"] = new(DD2.Log2, Math.Log2, 1.7),
            ["Log10"] = new(DD2.Log10, Math.Log10, 1.7),
            ["LogP1"] = new(DD2.LogP1, t => Math.Log(1.0 + t), 0.7),
            ["Sinh"] = new(DD2.Sinh, Math.Sinh, 0.7),
            ["Cosh"] = new(DD2.Cosh, Math.Cosh, 0.7),
            ["Tanh"] = new(DD2.Tanh, Math.Tanh, 0.7),
            ["Asinh"] = new(DD2.Asinh, Math.Asinh, 0.7),
            ["Acosh"] = new(DD2.Acosh, Math.Acosh, 1.7),
            ["Atanh"] = new(DD2.Atanh, Math.Atanh, 0.37),
            ["Sqrt"] = new(DD2.Sqrt, Math.Sqrt, 1.7),
            ["Cbrt"] = new(DD2.Cbrt, Math.Cbrt, 1.7),
            ["RootN(5)"] = new(a => DD2.RootN(a, 5), t => Math.Pow(t, 1.0 / 5.0), 1.7),
            ["RootN(-3)"] = new(a => DD2.RootN(a, -3), t => Math.Pow(t, -1.0 / 3.0), 1.7),
            ["Abs(positive)"] = new(DD2.Abs, Math.Abs, 1.7),
            ["Abs(negative)"] = new(DD2.Abs, Math.Abs, -1.7),
            ["Negate"] = new(a => -a, t => -t, 0.7),
            ["Pow(fixed 2.5)"] = new(a => DD2.Pow(a, DD2.Constant(2.5)), t => Math.Pow(t, 2.5), 1.7),
        };

        private static readonly Dictionary<string, BinaryCase> BinaryCases = new()
        {
            ["Add"] = new((a, b) => a + b, (u, v) => u + v, 0.7, 1.3),
            ["Subtract"] = new((a, b) => a - b, (u, v) => u - v, 0.7, 1.3),
            ["Multiply"] = new((a, b) => a * b, (u, v) => u * v, 0.7, 1.3),
            ["Divide"] = new((a, b) => a / b, (u, v) => u / v, 0.7, 1.3),
            ["Atan2"] = new(DD2.Atan2, Math.Atan2, 0.7, 1.3),
            ["Atan2 (negative x)"] = new(DD2.Atan2, Math.Atan2, 0.7, -1.3),
            ["Atan2Pi"] = new(DD2.Atan2Pi, (u, v) => Math.Atan2(u, v) / Math.PI, 0.7, 1.3),
            ["Hypot"] = new(DD2.Hypot, (u, v) => Math.Sqrt(u * u + v * v), 0.7, 1.3),
            ["Pow"] = new(DD2.Pow, Math.Pow, 1.7, 1.3),
            ["Log(x, base)"] = new(DD2.Log, (u, v) => Math.Log(u) / Math.Log(v), 1.7, 3.1),
            ["Modulo (both positive)"] = new((a, b) => a % b, (u, v) => u % v, 5.3, 1.3),
            ["Modulo (negative dividend)"] = new((a, b) => a % b, (u, v) => u % v, -5.3, 1.3),
            ["Modulo (negative divisor)"] = new((a, b) => a % b, (u, v) => u % v, 5.3, -1.3),
        };

        public static TheoryData<string> UnaryNames => Names(UnaryCases.Keys);
        public static TheoryData<string> BinaryNames => Names(BinaryCases.Keys);

        private static TheoryData<string> Names(IEnumerable<string> keys)
        {
            var data = new TheoryData<string>();
            foreach (string key in keys) data.Add(key);
            return data;
        }

        [Theory]
        [MemberData(nameof(UnaryNames))]
        public void UnaryFunction_ValueGradientHessian_MatchFiniteDifferences(string name)
        {
            UnaryCase c = UnaryCases[name];
            double s = c.Target / RawA0;

            double Reference(double x, double y) => c.Scalar(s * RawA(x, y));

            var (x, y) = DD2.Variables(X0, Y0);
            DD2 result = c.Ad(s * RawA(x, y));

            AssertMatchesFiniteDifferences(name, Reference, result);
        }

        [Theory]
        [MemberData(nameof(BinaryNames))]
        public void BinaryFunction_ValueGradientHessian_MatchFiniteDifferences(string name)
        {
            BinaryCase c = BinaryCases[name];
            double sa = c.TargetA / RawA0;
            double sb = c.TargetB / RawB0;

            double Reference(double x, double y) => c.Scalar(sa * RawA(x, y), sb * RawB(x, y));

            var (x, y) = DD2.Variables(X0, Y0);
            DD2 result = c.Ad(sa * RawA(x, y), sb * RawB(x, y));

            AssertMatchesFiniteDifferences(name, Reference, result);
        }

        private static void AssertMatchesFiniteDifferences(string name, Func<double, double, double> reference, in DD2 result)
        {
            Close(name + " / value", reference(X0, Y0), result.Value);
            Close(name + " / df/dx", NumericDiff.D1(reference, X0, Y0, 0), result.G(0));
            Close(name + " / df/dy", NumericDiff.D1(reference, X0, Y0, 1), result.G(1));
            Close(name + " / d2f/dx2", NumericDiff.D2(reference, X0, Y0, 0), result.H(0, 0));
            Close(name + " / d2f/dy2", NumericDiff.D2(reference, X0, Y0, 1), result.H(1, 1));
            Close(name + " / d2f/dxdy", NumericDiff.D2Mixed(reference, X0, Y0), result.H(0, 1));

            // The Hessian storage is triangular; both accessor orders must resolve to the same slot.
            Assert.Equal(result.H(0, 1), result.H(1, 0));
        }

        private static void Close(string what, double expected, double actual)
        {
            double tolerance = 1e-6 * (1.0 + Math.Abs(expected));
            Assert.True(Math.Abs(expected - actual) <= tolerance,
                $"{what}: expected {expected:R}, got {actual:R} (delta {Math.Abs(expected - actual):R}, tolerance {tolerance:R})");
        }
    }
}
