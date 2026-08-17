using System;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Algebraic identities and the zero-allocation claims from the README.
    /// </summary>
    /// <remarks>
    /// Identities are a strong check because they must hold in the derivatives too: if
    /// <c>f(x) == g(x)</c> for all <c>x</c> then their gradients and Hessians agree as well, so an
    /// error in one of the two derivative formulas shows up as a non-zero residual.
    /// </remarks>
    public class IdentityAndAllocationTests
    {
        private const double Tolerance = 1e-11;

        private static void AssertConstant(string what, DD2 f, double expectedValue)
        {
            Assert.True(Math.Abs(f.Value - expectedValue) < Tolerance, $"{what}: value {f.Value:R} != {expectedValue:R}");
            for (int i = 0; i < 2; i++)
            {
                Assert.True(Math.Abs(f.G(i)) < Tolerance, $"{what}: G({i}) = {f.G(i):R}, expected 0");
                for (int j = 0; j < 2; j++)
                {
                    Assert.True(Math.Abs(f.H(i, j)) < Tolerance, $"{what}: H({i},{j}) = {f.H(i, j):R}, expected 0");
                }
            }
        }

        private static void AssertIdentical(string what, DD2 a, DD2 b)
        {
            double scale = 1.0 + Math.Abs(a.Value);
            Assert.True(Math.Abs(a.Value - b.Value) < Tolerance * scale, $"{what}: values {a.Value:R} vs {b.Value:R}");
            for (int i = 0; i < 2; i++)
            {
                Assert.True(Math.Abs(a.G(i) - b.G(i)) < Tolerance * scale, $"{what}: G({i}) {a.G(i):R} vs {b.G(i):R}");
                for (int j = 0; j < 2; j++)
                {
                    Assert.True(Math.Abs(a.H(i, j) - b.H(i, j)) < Tolerance * scale, $"{what}: H({i},{j}) {a.H(i, j):R} vs {b.H(i, j):R}");
                }
            }
        }

        [Fact]
        public void PythagoreanIdentity_IsConstantInAllDerivatives()
        {
            var (x, y) = DD2.Variables(0.7, 1.3);
            DD2 a = 1.4 * x - 0.6 * y + 0.5 * x * y;

            AssertConstant("sin^2 + cos^2", DD2.Sin(a) * DD2.Sin(a) + DD2.Cos(a) * DD2.Cos(a), 1.0);
            AssertConstant("cosh^2 - sinh^2", DD2.Cosh(a) * DD2.Cosh(a) - DD2.Sinh(a) * DD2.Sinh(a), 1.0);
        }

        [Fact]
        public void InverseFunctionPairs_CancelInAllDerivatives()
        {
            var (x, y) = DD2.Variables(0.7, 1.3);
            DD2 positive = 0.4 + 0.3 * x + 0.2 * y + 0.1 * x * y; // stays in (0, 1)
            DD2 anyValue = 1.4 * x - 0.6 * y + 0.5 * x * y;

            AssertIdentical("log(exp(a))", DD2.Log(DD2.Exp(anyValue)), anyValue);
            AssertIdentical("exp(log(a))", DD2.Exp(DD2.Log(positive)), positive);
            AssertIdentical("sqrt(a)^2", DD2.Sqrt(positive) * DD2.Sqrt(positive), positive);
            AssertIdentical("cbrt(a)^3", DD2.Cbrt(positive) * DD2.Cbrt(positive) * DD2.Cbrt(positive), positive);
            AssertIdentical("asin(sin(a))", DD2.Asin(DD2.Sin(positive)), positive);
            AssertIdentical("atan(tan(a))", DD2.Atan(DD2.Tan(positive)), positive);
            AssertIdentical("atanh(tanh(a))", DD2.Atanh(DD2.Tanh(anyValue)), anyValue);
            AssertIdentical("asinh(sinh(a))", DD2.Asinh(DD2.Sinh(anyValue)), anyValue);
            AssertIdentical("expm1/logp1", DD2.LogP1(DD2.ExpM1(anyValue)), anyValue);
            AssertIdentical("rootN(a^3, 3)", DD2.RootN(positive * positive * positive, 3), positive);
        }

        [Fact]
        public void EquivalentFormulations_AgreeInAllDerivatives()
        {
            var (x, y) = DD2.Variables(0.7, 1.3);
            DD2 positive = 1.4 + 0.3 * x + 0.2 * y + 0.1 * x * y;
            DD2 other = 0.9 + 0.25 * x - 0.15 * y;

            AssertIdentical("a/b vs a*(1/b)", positive / other, positive * (DD2.One / other));
            AssertIdentical("pow(a,3) vs a*a*a", DD2.Pow(positive, DD2.Constant(3.0)), positive * positive * positive);
            AssertIdentical("sqrt(a) vs pow(a,0.5)", DD2.Sqrt(positive), DD2.Pow(positive, DD2.Constant(0.5)));
            AssertIdentical("exp2 vs pow(2,a)", DD2.Exp2(other), DD2.Pow(DD2.Constant(2.0), other));
            AssertIdentical("exp10 vs pow(10,a)", DD2.Exp10(other), DD2.Pow(DD2.Constant(10.0), other));
            AssertIdentical("log2 vs log/log2", DD2.Log2(positive), DD2.Log(positive) / DD2.Log(DD2.Constant(2.0)));
            AssertIdentical("log10 vs log/log10", DD2.Log10(positive), DD2.Log(positive) / DD2.Log(DD2.Constant(10.0)));
            AssertIdentical("hypot vs sqrt(a^2+b^2)", DD2.Hypot(positive, other), DD2.Sqrt(positive * positive + other * other));
            AssertIdentical("sinPi vs sin(pi*a)", DD2.SinPi(other), DD2.Sin(DD2.Pi * other));
            AssertIdentical("tanPi vs tan(pi*a)", DD2.TanPi(other), DD2.Tan(DD2.Pi * other));
            AssertIdentical("atanPi vs atan(a)/pi", DD2.AtanPi(other), DD2.Atan(other) / DD2.Pi);
            AssertIdentical("tanh vs sinh/cosh", DD2.Tanh(other), DD2.Sinh(other) / DD2.Cosh(other));
            AssertIdentical("atan2(a,1) vs atan(a)", DD2.Atan2(other, DD2.One), DD2.Atan(other));

            var (sin, cos) = DD2.SinCos(other);
            AssertIdentical("SinCos.Sin", sin, DD2.Sin(other));
            AssertIdentical("SinCos.Cos", cos, DD2.Cos(other));
        }

        /// <summary>
        /// <see cref="HyperJetMath"/> carries a second, hand-written implementation of the math
        /// functions for <c>DDScalar2</c> that lives alongside the generated generic-math members.
        /// The two must not drift apart.
        /// </summary>
        [Fact]
        public void HyperJetMath_AgreesWithTheGeneratedGenericMathMembers()
        {
            var (x, y) = DD2.Variables(0.7, 1.3);
            DD2 positive = 1.4 + 0.3 * x + 0.2 * y + 0.1 * x * y;
            DD2 small = 0.4 + 0.1 * x + 0.05 * y;

            AssertIdentical("Sin", Sin(small), DD2.Sin(small));
            AssertIdentical("Cos", Cos(small), DD2.Cos(small));
            AssertIdentical("Tan", Tan(small), DD2.Tan(small));
            AssertIdentical("Asin", Asin(small), DD2.Asin(small));
            AssertIdentical("Acos", Acos(small), DD2.Acos(small));
            AssertIdentical("Atan", Atan(small), DD2.Atan(small));
            AssertIdentical("Atan2", Atan2(small, positive), DD2.Atan2(small, positive));
            AssertIdentical("Exp", Exp(small), DD2.Exp(small));
            AssertIdentical("Log", Log(positive), DD2.Log(positive));
            AssertIdentical("Log10", Log10(positive), DD2.Log10(positive));
            AssertIdentical("Sqrt", Sqrt(positive), DD2.Sqrt(positive));
            AssertIdentical("Pow", Pow(positive, 2.5), DD2.Pow(positive, DD2.Constant(2.5)));
            AssertIdentical("Hypot", Hypot(small, positive), DD2.Hypot(small, positive));
            AssertIdentical("Abs", Abs(-positive), DD2.Abs(-positive));
        }

        /// <remarks>
        /// The measurement asks whether the <em>marginal</em> allocation is zero, not whether the
        /// first measured block is. Entering a hot loop for the first time costs a couple of
        /// kilobytes of one-time runtime bookkeeping (tiered JIT, on-stack replacement) that
        /// <see cref="GC.GetAllocatedBytesForCurrentThread"/> attributes to this thread, and how much
        /// of it lands inside the measured window depends on the machine. A genuine per-operation
        /// allocation would show up in every block, so requiring one block of exactly zero bytes is
        /// both strict and immune to JIT timing.
        /// </remarks>
        [Fact]
        public void DDScalarSpan_ZeroAllocationMethods_DoNotAllocate()
        {
            const int size = 4, order = 2;
            int n = Kernel.GetDataLength(size, order);

            Span<double> aBuffer = stackalloc double[n];
            Span<double> bBuffer = stackalloc double[n];
            Span<double> destBuffer = stackalloc double[n];

            var a = DDScalarSpan.Variable(aBuffer, 0, 1.3, size, order);
            var b = DDScalarSpan.Variable(bBuffer, 1, 2.7, size, order);
            var dest = new DDScalarSpan(destBuffer, size, order);

            long[] measured = new long[Attempts];

            for (int attempt = 0; attempt < Attempts; attempt++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();

                for (int i = 0; i < BlockSize; i++)
                {
                    a.Multiply(b, dest);
                    a.Divide(b, dest);
                    a.Sin(dest);
                    a.Exp(dest);
                    a.Sqrt(dest);
                    a.Hypot(b, dest);
                }

                measured[attempt] = GC.GetAllocatedBytesForCurrentThread() - before;
                if (measured[attempt] == 0) return;
            }

            Assert.Fail(Diagnosis("DDScalarSpan", measured));
        }

        [Fact]
        public void DDScalarN_Arithmetic_DoesNotAllocate()
        {
            var (x, y) = DD2.Variables(3.0, 6.0);
            long[] measured = new long[Attempts];

            for (int attempt = 0; attempt < Attempts; attempt++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();

                for (int i = 0; i < BlockSize; i++) Consume((x * y) / (x - y) + DD2.Sin(x) * DD2.Exp(y));

                measured[attempt] = GC.GetAllocatedBytesForCurrentThread() - before;
                if (measured[attempt] == 0) return;
            }

            Assert.Fail(Diagnosis("DDScalar2<double>", measured));
        }

        private const int BlockSize = 1000;
        private const int Attempts = 10;

        private static string Diagnosis(string what, long[] measured) =>
            $"{what} kept allocating: {Attempts} blocks of {BlockSize} iterations measured " +
            $"{string.Join(", ", measured)} bytes, none of them zero.";

        private static double _sink;

        private static void Consume(DD2 value) => _sink = value.Value;
    }
}
