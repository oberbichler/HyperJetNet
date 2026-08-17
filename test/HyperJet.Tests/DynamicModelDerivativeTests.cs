using System;
using System.Collections.Generic;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// The dynamic <see cref="DDScalar"/> and the ref-struct <see cref="DDScalarSpan"/> carry their own,
    /// hand-written copies of every derivative formula — independent of the generated
    /// <c>DDScalar{n}</c> code. These tests hold each copy against the same finite-difference
    /// ground truth so a typo in one model cannot hide behind a correct one.
    /// </summary>
    public class DynamicModelDerivativeTests
    {
        private const double X0 = 0.5;
        private const double Y0 = 0.4;
        private const int Size = 2;
        private const int Order = 2;

        private static double RawA(double x, double y) => 1.1 * x + 0.9 * y + 0.3 * x * y;
        private static DDScalar RawA(DDScalar x, DDScalar y) => 1.1 * x + 0.9 * y + 0.3 * x * y;
        private static double RawB(double x, double y) => 0.8 * x - 0.6 * y + 0.25 * x * y + 0.5;
        private static DDScalar RawB(DDScalar x, DDScalar y) => 0.8 * x - 0.6 * y + 0.25 * x * y + 0.5;

        private const double RawA0 = 1.1 * X0 + 0.9 * Y0 + 0.3 * X0 * Y0;
        private const double RawB0 = 0.8 * X0 - 0.6 * Y0 + 0.25 * X0 * Y0 + 0.5;

        private sealed record UnaryCase(Func<DDScalar, DDScalar> Ad, Func<double, double> Scalar, double Target);
        private sealed record BinaryCase(Func<DDScalar, DDScalar, DDScalar> Ad, Func<double, double, double> Scalar, double TargetA, double TargetB);

        private static readonly Dictionary<string, UnaryCase> UnaryCases = new()
        {
            ["Sin"] = new(a => Sin(a), Math.Sin, 0.7),
            ["Cos"] = new(a => Cos(a), Math.Cos, 0.7),
            ["Tan"] = new(a => Tan(a), Math.Tan, 0.7),
            ["Asin"] = new(a => Asin(a), Math.Asin, 0.37),
            ["Acos"] = new(a => Acos(a), Math.Acos, 0.37),
            ["Atan"] = new(a => Atan(a), Math.Atan, 0.7),
            ["Exp"] = new(a => Exp(a), Math.Exp, 0.7),
            ["Log"] = new(a => Log(a), Math.Log, 1.7),
            ["Log2"] = new(a => Log2(a), Math.Log2, 1.7),
            ["Log10"] = new(a => Log10(a), Math.Log10, 1.7),
            ["Sqrt"] = new(a => Sqrt(a), Math.Sqrt, 1.7),
            ["Cbrt"] = new(a => Cbrt(a), Math.Cbrt, 1.7),
            ["Sinh"] = new(a => Sinh(a), Math.Sinh, 0.7),
            ["Cosh"] = new(a => Cosh(a), Math.Cosh, 0.7),
            ["Tanh"] = new(a => Tanh(a), Math.Tanh, 0.7),
            ["Pow(2.5)"] = new(a => Pow(a, 2.5), t => Math.Pow(t, 2.5), 1.7),
            ["SinPi"] = new(a => SinPi(a), t => Math.Sin(Math.PI * t), 0.37),
            ["CosPi"] = new(a => CosPi(a), t => Math.Cos(Math.PI * t), 0.37),
            ["TanPi"] = new(a => TanPi(a), t => Math.Tan(Math.PI * t), 0.37),
            ["AsinPi"] = new(a => AsinPi(a), t => Math.Asin(t) / Math.PI, 0.37),
            ["AcosPi"] = new(a => AcosPi(a), t => Math.Acos(t) / Math.PI, 0.37),
            ["AtanPi"] = new(a => AtanPi(a), t => Math.Atan(t) / Math.PI, 0.7),
            ["Exp2"] = new(a => Exp2(a), t => Math.Pow(2.0, t), 0.7),
            ["Exp10"] = new(a => Exp10(a), t => Math.Pow(10.0, t), 0.7),
            ["ExpM1"] = new(a => ExpM1(a), t => Math.Exp(t) - 1.0, 0.7),
            ["LogP1"] = new(a => LogP1(a), t => Math.Log(1.0 + t), 0.7),
            ["Asinh"] = new(a => Asinh(a), Math.Asinh, 0.7),
            ["Acosh"] = new(a => Acosh(a), Math.Acosh, 1.7),
            ["Atanh"] = new(a => Atanh(a), Math.Atanh, 0.37),
            ["RootN(5)"] = new(a => RootN(a, 5), t => Math.Pow(t, 1.0 / 5.0), 1.7),
            ["RootN(-3)"] = new(a => RootN(a, -3), t => Math.Pow(t, -1.0 / 3.0), 1.7),
            ["Abs(positive)"] = new(a => Abs(a), Math.Abs, 1.7),
            ["Abs(negative)"] = new(a => Abs(a), Math.Abs, -1.7),
            ["Negate"] = new(a => -a, t => -t, 0.7),
            ["ScalarDivideInto"] = new(a => 3.0 / a, t => 3.0 / t, 1.7),
            ["ScalarSubtractFrom"] = new(a => 3.0 - a, t => 3.0 - t, 1.7),
            ["ScalarMultiply"] = new(a => a * 2.5, t => t * 2.5, 0.7),
            ["ScalarDivide"] = new(a => a / 2.5, t => t / 2.5, 0.7),
            ["ScalarAdd"] = new(a => a + 2.5, t => t + 2.5, 0.7),
            ["ScalarSubtract"] = new(a => a - 2.5, t => t - 2.5, 0.7),
        };

        private static readonly Dictionary<string, BinaryCase> BinaryCases = new()
        {
            ["Add"] = new((a, b) => a + b, (u, v) => u + v, 0.7, 1.3),
            ["Subtract"] = new((a, b) => a - b, (u, v) => u - v, 0.7, 1.3),
            ["Multiply"] = new((a, b) => a * b, (u, v) => u * v, 0.7, 1.3),
            ["Divide"] = new((a, b) => a / b, (u, v) => u / v, 0.7, 1.3),
            ["Atan2"] = new((a, b) => Atan2(a, b), Math.Atan2, 0.7, 1.3),
            ["Atan2Pi"] = new((a, b) => Atan2Pi(a, b), (u, v) => Math.Atan2(u, v) / Math.PI, 0.7, 1.3),
            ["Hypot"] = new((a, b) => Hypot(a, b), (u, v) => Math.Sqrt(u * u + v * v), 0.7, 1.3),
            ["Pow"] = new((a, b) => Pow(a, b), Math.Pow, 1.7, 1.3),
            ["Log(x, base)"] = new((a, b) => Log(a, b), (u, v) => Math.Log(u) / Math.Log(v), 1.7, 3.1),
        };

        public static TheoryData<string> UnaryNames => Names(UnaryCases.Keys);
        public static TheoryData<string> BinaryNames => Names(BinaryCases.Keys);
        public static TheoryData<string> SpanUnaryNames => Names(SpanUnary.Keys);

        private static TheoryData<string> Names(IEnumerable<string> keys)
        {
            var data = new TheoryData<string>();
            foreach (string key in keys) data.Add(key);
            return data;
        }

        [Theory]
        [MemberData(nameof(UnaryNames))]
        public void DDScalar_UnaryFunction_MatchesFiniteDifferences(string name)
        {
            UnaryCase c = UnaryCases[name];
            double s = c.Target / RawA0;

            double Reference(double x, double y) => c.Scalar(s * RawA(x, y));

            var (x, y) = DDScalar.Variables(new[] { X0, Y0 }, Order);
            DDScalar result = c.Ad(s * RawA(x, y));

            AssertMatches(name, Reference, result);
        }

        [Theory]
        [MemberData(nameof(BinaryNames))]
        public void DDScalar_BinaryFunction_MatchesFiniteDifferences(string name)
        {
            BinaryCase c = BinaryCases[name];
            double sa = c.TargetA / RawA0;
            double sb = c.TargetB / RawB0;

            double Reference(double x, double y) => c.Scalar(sa * RawA(x, y), sb * RawB(x, y));

            var (x, y) = DDScalar.Variables(new[] { X0, Y0 }, Order);
            DDScalar result = c.Ad(sa * RawA(x, y), sb * RawB(x, y));

            AssertMatches(name, Reference, result);
        }

        // DDScalarSpan's transcendent methods are instance methods writing into a destination span,
        // so they are driven through a delegate that receives already-allocated buffers.
        private delegate void SpanOp(in DDScalarSpan a, DDScalarSpan destination);

        private static readonly Dictionary<string, (SpanOp Op, Func<double, double> Scalar, double Target)> SpanUnary = new()
        {
            ["Sin"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Sin(d), Math.Sin, 0.7),
            ["Cos"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Cos(d), Math.Cos, 0.7),
            ["Tan"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Tan(d), Math.Tan, 0.7),
            ["Asin"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Asin(d), Math.Asin, 0.37),
            ["Acos"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Acos(d), Math.Acos, 0.37),
            ["Atan"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Atan(d), Math.Atan, 0.7),
            ["Exp"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Exp(d), Math.Exp, 0.7),
            ["Log"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Log(d), Math.Log, 1.7),
            ["Log2"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Log2(d), Math.Log2, 1.7),
            ["Log10"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Log10(d), Math.Log10, 1.7),
            ["Sqrt"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Sqrt(d), Math.Sqrt, 1.7),
            ["Cbrt"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Cbrt(d), Math.Cbrt, 1.7),
            ["Sinh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Sinh(d), Math.Sinh, 0.7),
            ["Cosh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Cosh(d), Math.Cosh, 0.7),
            ["Tanh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Tanh(d), Math.Tanh, 0.7),
            ["Pow(2.5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Pow(2.5, d), t => Math.Pow(t, 2.5), 1.7),
            ["Abs(positive)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Abs(d), Math.Abs, 1.7),
            ["Abs(negative)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Abs(d), Math.Abs, -1.7),
            ["Negate"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Negate(d), t => -t, 0.7),
            ["DivideInto(3)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.DivideInto(3.0, d), t => 3.0 / t, 1.7),
            ["SubtractFrom(3)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.SubtractFrom(3.0, d), t => 3.0 - t, 1.7),
            ["Multiply(2.5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Multiply(2.5, d), t => t * 2.5, 0.7),
            ["Divide(2.5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Divide(2.5, d), t => t / 2.5, 0.7),
            ["Add(2.5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Add(2.5, d), t => t + 2.5, 0.7),
            ["Subtract(2.5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Subtract(2.5, d), t => t - 2.5, 0.7),
            ["SinPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.SinPi(d), t => Math.Sin(Math.PI * t), 0.37),
            ["CosPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.CosPi(d), t => Math.Cos(Math.PI * t), 0.37),
            ["TanPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.TanPi(d), t => Math.Tan(Math.PI * t), 0.37),
            ["AsinPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.AsinPi(d), t => Math.Asin(t) / Math.PI, 0.37),
            ["AcosPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.AcosPi(d), t => Math.Acos(t) / Math.PI, 0.37),
            ["AtanPi"] = ((in DDScalarSpan a, DDScalarSpan d) => a.AtanPi(d), t => Math.Atan(t) / Math.PI, 0.7),
            ["Exp2"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Exp2(d), t => Math.Pow(2.0, t), 0.7),
            ["Exp10"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Exp10(d), t => Math.Pow(10.0, t), 0.7),
            ["ExpM1"] = ((in DDScalarSpan a, DDScalarSpan d) => a.ExpM1(d), t => Math.Exp(t) - 1.0, 0.7),
            ["LogP1"] = ((in DDScalarSpan a, DDScalarSpan d) => a.LogP1(d), t => Math.Log(1.0 + t), 0.7),
            ["Asinh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Asinh(d), Math.Asinh, 0.7),
            ["Acosh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Acosh(d), Math.Acosh, 1.7),
            ["Atanh"] = ((in DDScalarSpan a, DDScalarSpan d) => a.Atanh(d), Math.Atanh, 0.37),
            ["RootN(5)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.RootN(5, d), t => Math.Pow(t, 1.0 / 5.0), 1.7),
            ["RootN(-3)"] = ((in DDScalarSpan a, DDScalarSpan d) => a.RootN(-3, d), t => Math.Pow(t, -1.0 / 3.0), 1.7),
        };

        [Theory]
        [MemberData(nameof(SpanUnaryNames))]
        public void DDScalarSpan_UnaryFunction_MatchesFiniteDifferences(string name)
        {
            var (op, scalar, target) = SpanUnary[name];
            double s = target / RawA0;

            double Reference(double x, double y) => scalar(s * RawA(x, y));

            // Build the inner expression via the dynamic model, then hand its raw data to a span.
            var (dx, dy) = DDScalar.Variables(new[] { X0, Y0 }, Order);
            DDScalar innerDyn = s * RawA(dx, dy);

            var inner = new DDScalarSpan(innerDyn.AsSpan(), Size, Order);
            Span<double> destBuffer = stackalloc double[Kernel.GetDataLength(Size, Order)];
            var dest = new DDScalarSpan(destBuffer, Size, Order);

            op(inner, dest);

            AssertMatches(name, Reference, dest.Value, dest.G(0), dest.G(1), dest.H(0, 0), dest.H(1, 1), dest.H(0, 1), dest.H(1, 0));
        }

        [Fact]
        public void DDScalarSpan_BinaryFunctions_MatchFiniteDifferences()
        {
            const double sa = 0.7 / RawA0;
            const double sb = 1.3 / RawB0;

            var (dx, dy) = DDScalar.Variables(new[] { X0, Y0 }, Order);
            DDScalar aDyn = sa * RawA(dx, dy);
            DDScalar bDyn = sb * RawB(dx, dy);

            var a = new DDScalarSpan(aDyn.AsSpan(), Size, Order);
            var b = new DDScalarSpan(bDyn.AsSpan(), Size, Order);

            int n = Kernel.GetDataLength(Size, Order);
            Span<double> buffer = stackalloc double[n];
            var dest = new DDScalarSpan(buffer, Size, Order);

            // dest is a ref struct and cannot be captured by the local function, so its
            // derivatives are snapshotted into plain doubles before each comparison.
            static void Check(string name, Func<double, double, double> scalar, double[] snapshot)
            {
                double Reference(double x, double y) => scalar(sa * RawA(x, y), sb * RawB(x, y));
                AssertMatches(name, Reference, snapshot[0], snapshot[1], snapshot[2], snapshot[3], snapshot[4], snapshot[5], snapshot[6]);
            }

            a.Add(b, dest); Check("Add", (u, v) => u + v, Snapshot(dest));
            a.Subtract(b, dest); Check("Subtract", (u, v) => u - v, Snapshot(dest));
            a.Multiply(b, dest); Check("Multiply", (u, v) => u * v, Snapshot(dest));
            a.Divide(b, dest); Check("Divide", (u, v) => u / v, Snapshot(dest));
            a.Atan2(b, dest); Check("Atan2", Math.Atan2, Snapshot(dest));
            a.Atan2Pi(b, dest); Check("Atan2Pi", (u, v) => Math.Atan2(u, v) / Math.PI, Snapshot(dest));
            a.Hypot(b, dest); Check("Hypot", (u, v) => Math.Sqrt(u * u + v * v), Snapshot(dest));
            a.Pow(b, dest); Check("Pow", Math.Pow, Snapshot(dest));
            a.Log(b, dest); Check("Log(x, base)", (u, v) => Math.Log(u) / Math.Log(v), Snapshot(dest));
        }

        [Fact]
        public void DDScalarSpan_SinCosPairs_MatchTheIndividualFunctions()
        {
            const double s = 0.37 / RawA0;

            var (dx, dy) = DDScalar.Variables(new[] { X0, Y0 }, Order);
            DDScalar innerDyn = s * RawA(dx, dy);
            var inner = new DDScalarSpan(innerDyn.AsSpan(), Size, Order);

            int n = Kernel.GetDataLength(Size, Order);
            Span<double> pool = stackalloc double[2 * n];
            var sinDest = new DDScalarSpan(pool[..n], Size, Order);
            var cosDest = new DDScalarSpan(pool[n..], Size, Order);

            inner.SinCos(sinDest, cosDest);
            AssertMatches("SinCos.Sin", (x, y) => Math.Sin(s * RawA(x, y)), Snapshot(sinDest));
            AssertMatches("SinCos.Cos", (x, y) => Math.Cos(s * RawA(x, y)), Snapshot(cosDest));

            inner.SinCosPi(sinDest, cosDest);
            AssertMatches("SinCosPi.Sin", (x, y) => Math.Sin(Math.PI * s * RawA(x, y)), Snapshot(sinDest));
            AssertMatches("SinCosPi.Cos", (x, y) => Math.Cos(Math.PI * s * RawA(x, y)), Snapshot(cosDest));
        }

        private static void AssertMatches(string name, Func<double, double, double> reference, double[] snapshot)
        {
            AssertMatches(name, reference, snapshot[0], snapshot[1], snapshot[2], snapshot[3], snapshot[4], snapshot[5], snapshot[6]);
        }

        /// <summary>Value, gradient and Hessian of a ref struct as plain doubles, so lambdas can use them.</summary>
        private static double[] Snapshot(in DDScalarSpan s) => new[]
        {
            s.Value, s.G(0), s.G(1), s.H(0, 0), s.H(1, 1), s.H(0, 1), s.H(1, 0),
        };

        private static void AssertMatches(string name, Func<double, double, double> reference, in DDScalar result)
        {
            AssertMatches(name, reference, result.Value, result.G(0), result.G(1),
                result.H(0, 0), result.H(1, 1), result.H(0, 1), result.H(1, 0));
        }

        private static void AssertMatches(string name, Func<double, double, double> reference,
            double value, double gx, double gy, double hxx, double hyy, double hxy, double hyx)
        {
            Close(name + " / value", reference(X0, Y0), value);
            Close(name + " / df/dx", NumericDiff.D1(reference, X0, Y0, 0), gx);
            Close(name + " / df/dy", NumericDiff.D1(reference, X0, Y0, 1), gy);
            Close(name + " / d2f/dx2", NumericDiff.D2(reference, X0, Y0, 0), hxx);
            Close(name + " / d2f/dy2", NumericDiff.D2(reference, X0, Y0, 1), hyy);
            Close(name + " / d2f/dxdy", NumericDiff.D2Mixed(reference, X0, Y0), hxy);
            Assert.Equal(hxy, hyx);
        }

        private static void Close(string what, double expected, double actual)
        {
            double tolerance = 1e-6 * (1.0 + Math.Abs(expected));
            Assert.True(Math.Abs(expected - actual) <= tolerance,
                $"{what}: expected {expected:R}, got {actual:R} (delta {Math.Abs(expected - actual):R}, tolerance {tolerance:R})");
        }
    }
}
