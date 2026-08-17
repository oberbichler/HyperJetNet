using System;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Uses the free-function spelling the README advertises — <c>using static HyperJet.HyperJetMath;</c>
    /// then <c>Sin(x)</c> — against several dimensions and both backing types. This is a compile-time
    /// proof that the facade really covers more than one dimension; the reflection sweep in
    /// <see cref="FunctionParityTests"/> only checks that the names exist.
    /// </summary>
    public class FacadeUsageTests
    {
        [Fact]
        public void Dimension1_ThroughTheFacade()
        {
            var x = DDScalar1<double>.Variables(0.7);
            DDScalar1<double> f = Sin(x) + Exp2(x) + Atanh(x * 0.5);

            Assert.Equal(Math.Sin(0.7) + Math.Pow(2.0, 0.7) + Math.Atanh(0.35), f.Value, precision: 12);
            Assert.Equal(DDScalar1<double>.Sin(x).G(0) + DDScalar1<double>.Exp2(x).G(0)
                       + DDScalar1<double>.Atanh(x * 0.5).G(0), f.G(0), precision: 12);
        }

        [Fact]
        public void Dimension3_ThroughTheFacade()
        {
            var (x, y, z) = DDScalar3<double>.Variables(0.7, 1.3, 0.4);

            DDScalar3<double> f = Hypot(x, y) * Cos(z) + LogP1(x * z) - RootN(y, 3);

            DDScalar3<double> expected = DDScalar3<double>.Hypot(x, y) * DDScalar3<double>.Cos(z)
                                       + DDScalar3<double>.LogP1(x * z)
                                       - DDScalar3<double>.RootN(y, 3);

            Assert.Equal(expected.Value, f.Value, precision: 12);
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(expected.G(i), f.G(i), precision: 12);
                for (int j = 0; j < 3; j++) Assert.Equal(expected.H(i, j), f.H(i, j), precision: 12);
            }
        }

        [Fact]
        public void Dimension7_ThroughTheFacade()
        {
            var v = DDScalar7<double>.Variables(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8);

            DDScalar7<double> f = Sinh(v.v1) + Acosh(v.v2 + 1.5) + Atan2(v.v3, v.v4)
                                + Cbrt(v.v5) + SinPi(v.v6) + ExpM1(v.v7);

            Assert.True(double.IsFinite(f.Value));
            Assert.Equal(Math.Cosh(0.2), f.G(0), precision: 12);
            Assert.Equal(Math.Sinh(0.2), f.H(0, 0), precision: 12);
            Assert.Equal(0.0, f.H(0, 1), precision: 12); // the terms are separable
        }

        [Fact]
        public void Dimension15_ThroughTheFacade()
        {
            var v = DDScalar15<double>.Variables(
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5);

            DDScalar15<double> f = Tanh(v.v1) * Log10(v.v15) + Sqrt(v.v8) - Exp10(v.v2);

            DDScalar15<double> expected = DDScalar15<double>.Tanh(v.v1) * DDScalar15<double>.Log10(v.v15)
                                        + DDScalar15<double>.Sqrt(v.v8)
                                        - DDScalar15<double>.Exp10(v.v2);

            Assert.Equal(expected.Value, f.Value, precision: 12);
            Assert.Equal(expected.G(0), f.G(0), precision: 12);
            Assert.Equal(expected.H(0, 14), f.H(0, 14), precision: 12);
        }

        [Fact]
        public void FloatBacking_ThroughTheFacade()
        {
            var (x, y) = DDScalar4<float>.Variables(0.7f, 1.3f, 0.4f, 0.9f) is var v
                ? (v.x, v.y)
                : default;

            DDScalar4<float> f = Sin(x) * Cosh(y);

            Assert.Equal(MathF.Sin(0.7f) * MathF.Cosh(1.3f), f.Value, tolerance: 1e-5f);
            Assert.Equal(MathF.Cos(0.7f) * MathF.Cosh(1.3f), f.G(0), tolerance: 1e-5f);
        }

        [Fact]
        public void ConstantExponentPow_AcceptsANegativeBase()
        {
            // Pow(a, b) with an active exponent needs log(a); the constant-exponent overload does not.
            var x = DDScalar1<double>.Variables(-2.0);

            DDScalar1<double> cubed = Pow(x, 3.0);

            Assert.Equal(-8.0, cubed.Value, precision: 12);
            Assert.Equal(12.0, cubed.G(0), precision: 12);   // 3x^2
            Assert.Equal(-12.0, cubed.H(0, 0), precision: 12); // 6x
        }

        [Fact]
        public void SinCos_ThroughTheFacade()
        {
            var (x, y) = DDScalar2<double>.Variables(0.7, 1.3);

            var (sin, cos) = SinCos(x * y);

            Assert.Equal(Math.Sin(0.91), sin.Value, precision: 12);
            Assert.Equal(Math.Cos(0.91), cos.Value, precision: 12);
        }

        [Fact]
        public void DynamicModel_UsesTheSameSpelling()
        {
            // The point of the facade: the same source text works for the dynamic model too.
            var (x, y) = DDScalar.Variables(new[] { 0.7, 1.3 });
            DDScalar f = Sin(x) * Cosh(y);

            var (sx, sy) = DDScalar2<double>.Variables(0.7, 1.3);
            DDScalar2<double> expected = Sin(sx) * Cosh(sy);

            Assert.Equal(expected.Value, f.Value, precision: 12);
            Assert.Equal(expected.G(0), f.G(0), precision: 12);
            Assert.Equal(expected.H(0, 1), f.H(0, 1), precision: 12);
        }
    }
}
