using System;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using DD3 = HyperJet.DDScalar3<double>;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// <c>FusedMultiplyAdd</c> and <c>Ieee754Remainder</c>: the two members of
    /// <see cref="System.Math"/> that were missing and that carry a meaningful derivative.
    /// </summary>
    /// <remarks>
    /// Both are exactly representable — one is bilinear, the other piecewise linear — so the
    /// derivatives are asserted against their closed forms rather than against finite differences.
    /// The chain rule is covered separately by composing them with non-trivial inner expressions.
    /// </remarks>
    public class Ieee754OperationTests
    {
        #region FusedMultiplyAdd

        [Fact]
        public void FusedMultiplyAdd_HasTheExactBilinearDerivatives()
        {
            const double x0 = 1.7, y0 = -2.3, z0 = 0.9;
            var (x, y, z) = DD3.Variables(x0, y0, z0);

            DD3 f = DD3.FusedMultiplyAdd(x, y, z);

            Assert.Equal(Math.FusedMultiplyAdd(x0, y0, z0), f.Value);

            Assert.Equal(y0, f.G(0), precision: 12);   // d/dx = y
            Assert.Equal(x0, f.G(1), precision: 12);   // d/dy = x
            Assert.Equal(1.0, f.G(2), precision: 12);  // d/dz = 1

            // The only non-zero second derivative is the mixed d2/dxdy.
            double[,] h = f.GetHessian();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    double expected = (i == 0 && j == 1) || (i == 1 && j == 0) ? 1.0 : 0.0;
                    Assert.Equal(expected, h[i, j], precision: 12);
                }
        }

        /// <summary>
        /// The point of the function is the single rounding: at these inputs <c>x*y + z</c> and the
        /// fused form differ, and the AD result has to follow the fused one.
        /// </summary>
        [Fact]
        public void FusedMultiplyAdd_RoundsOnlyOnce()
        {
            double eps = Math.Pow(2, -52);
            double x0 = 1.0 + eps;
            double y0 = 1.0 - eps;
            const double z0 = -1.0;

            Assert.NotEqual(x0 * y0 + z0, Math.FusedMultiplyAdd(x0, y0, z0));

            var (x, y, z) = DD3.Variables(x0, y0, z0);

            Assert.Equal(Math.FusedMultiplyAdd(x0, y0, z0), DD3.FusedMultiplyAdd(x, y, z).Value);
        }

        /// <summary>
        /// Composed with inner expressions the result must match the plain product-plus-sum built
        /// from the operators, in every derivative. This is what exercises the ternary kernel.
        /// </summary>
        [Fact]
        public void FusedMultiplyAdd_ComposesLikeTheEquivalentExpression()
        {
            var (u, v) = DD2.Variables(0.6, 1.4);

            DD2 a = 1.1 * u - 0.4 * v + 0.3 * u * v;
            DD2 b = 0.7 * u + 0.9 * v - 0.2 * u * v;
            DD2 c = DD2.Sin(u) * DD2.Exp(0.5 * v);

            DD2 fused = DD2.FusedMultiplyAdd(a, b, c);
            DD2 plain = a * b + c;

            // Values may differ in the last bit by design; the derivatives must not.
            Assert.Equal(plain.Value, fused.Value, precision: 12);
            Assert.Equal(plain.G(0), fused.G(0), precision: 12);
            Assert.Equal(plain.G(1), fused.G(1), precision: 12);
            Assert.Equal(plain.H(0, 0), fused.H(0, 0), precision: 12);
            Assert.Equal(plain.H(0, 1), fused.H(0, 1), precision: 12);
            Assert.Equal(plain.H(1, 1), fused.H(1, 1), precision: 12);
        }

        public static TheoryData<int> Sizes
        {
            get
            {
                var data = new TheoryData<int>();
                for (int size = 1; size <= 12; size++) data.Add(size);
                return data;
            }
        }

        /// <summary>
        /// The dynamic models reach the vectorized ternary kernel, which had no caller in the library
        /// before this function existed. Sizes 1..12 give data lengths 3..91 and so cross every SIMD
        /// threshold; composing with inner expressions makes the second-order block contribute.
        /// </summary>
        [Theory]
        [MemberData(nameof(Sizes))]
        public void FusedMultiplyAdd_DynamicModel_ComposesLikeTheEquivalentExpression(int size)
        {
            const int order = 2;

            double[] point = new double[size];
            for (int i = 0; i < size; i++) point[i] = 0.3 + 0.07 * i;

            DDScalar[] v = DDScalar.Variables(point, order);

            DDScalar a = Inner(v, 1.1, -0.4, 0.3);
            DDScalar b = Inner(v, 0.7, 0.9, -0.2);
            DDScalar c = Inner(v, -0.5, 0.6, 0.15);

            DDScalar fused = FusedMultiplyAdd(a, b, c);
            DDScalar plain = a * b + c;

            AssertSameDerivatives($"dynamic FusedMultiplyAdd size={size}", plain, fused, size);

            // And the same through the zero-allocation model.
            Span<double> destBuffer = stackalloc double[Kernel.GetDataLength(size, order)];
            var dest = new DDScalarSpan(destBuffer, size, order);
            new DDScalarSpan(a.AsSpan(), size, order)
                .FusedMultiplyAdd(new DDScalarSpan(b.AsSpan(), size, order),
                                  new DDScalarSpan(c.AsSpan(), size, order), dest);

            for (int i = 0; i < size; i++)
            {
                Assert.True(Math.Abs(plain.G(i) - dest.G(i)) < 1e-11, $"span size={size} G({i})");
                for (int j = 0; j < size; j++)
                    Assert.True(Math.Abs(plain.H(i, j) - dest.H(i, j)) < 1e-11, $"span size={size} H({i},{j})");
            }
        }

        /// <summary>A quadratic combination of all variables, so gradient and Hessian are both dense.</summary>
        private static DDScalar Inner(DDScalar[] v, double linear, double cross, double square)
        {
            int n = v.Length;
            DDScalar result = DDScalar.Constant(0.5, n, v[0].Order);

            for (int i = 0; i < n; i++)
            {
                result += (linear + 0.05 * i) * v[i];
                result += square * v[i] * v[i];
                result += cross * v[i] * v[(i + 1) % n];
            }

            return result;
        }

        private static void AssertSameDerivatives(string what, in DDScalar expected, in DDScalar actual, int size)
        {
            Assert.True(Math.Abs(expected.Value - actual.Value) < 1e-11, $"{what} value");

            for (int i = 0; i < size; i++)
            {
                Assert.True(Math.Abs(expected.G(i) - actual.G(i)) < 1e-11, $"{what} G({i})");
                for (int j = 0; j < size; j++)
                    Assert.True(Math.Abs(expected.H(i, j) - actual.H(i, j)) < 1e-11, $"{what} H({i},{j})");
            }
        }

        [Fact]
        public void FusedMultiplyAdd_AgreesAcrossAllThreeModels()
        {
            double[] point = { 1.7, -2.3, 0.9 };
            const int size = 3, order = 2;

            var (sx, sy, sz) = DD3.Variables(point[0], point[1], point[2]);
            DD3 statik = DD3.FusedMultiplyAdd(sx, sy, sz);

            var d = DDScalar.Variables(point, order);
            DDScalar dynamik = FusedMultiplyAdd(d[0], d[1], d[2]);

            Span<double> destBuffer = stackalloc double[Kernel.GetDataLength(size, order)];
            var dest = new DDScalarSpan(destBuffer, size, order);
            new DDScalarSpan(d[0].AsSpan(), size, order)
                .FusedMultiplyAdd(new DDScalarSpan(d[1].AsSpan(), size, order),
                                  new DDScalarSpan(d[2].AsSpan(), size, order), dest);

            AssertSameDerivatives("FusedMultiplyAdd", statik, dynamik, dest, size);
        }

        #endregion

        #region Ieee754Remainder

        [Theory]
        [InlineData(5.3, 1.3)]
        [InlineData(-5.3, 1.3)]
        [InlineData(5.3, -1.3)]
        [InlineData(-5.3, -1.3)]
        [InlineData(0.7, 4.1)]
        [InlineData(17.0, 5.0)]
        public void Ieee754Remainder_MatchesMathAndHasTheExactDerivatives(double a0, double b0)
        {
            var (a, b) = DD2.Variables(a0, b0);

            DD2 f = DD2.Ieee754Remainder(a, b);

            Assert.Equal(Math.IEEERemainder(a0, b0), f.Value);

            // f = a - b*q with q = round(a/b) held fixed between break points.
            double q = Math.Round(a0 / b0, MidpointRounding.ToEven);
            Assert.Equal(1.0, f.G(0), precision: 12);
            Assert.Equal(-q, f.G(1), precision: 12);

            // Piecewise linear, so the Hessian vanishes.
            Assert.Equal(0.0, f.H(0, 0), precision: 12);
            Assert.Equal(0.0, f.H(0, 1), precision: 12);
            Assert.Equal(0.0, f.H(1, 1), precision: 12);
        }

        /// <summary>
        /// Round-to-nearest, not truncation: with a quotient just above one half the remainder turns
        /// negative and the derivative with respect to the divisor picks up the larger quotient.
        /// This is what separates it from the <c>%</c> operator.
        /// </summary>
        [Fact]
        public void Ieee754Remainder_DiffersFromTheModuloOperator()
        {
            const double a0 = 5.9, b0 = 1.0;
            var (a, b) = DD2.Variables(a0, b0);

            DD2 ieee = DD2.Ieee754Remainder(a, b);
            DD2 modulo = a % b;

            Assert.Equal(-0.1, ieee.Value, precision: 12);  // 5.9 - 1.0 * 6
            Assert.Equal(0.9, modulo.Value, precision: 12); // 5.9 - 1.0 * 5

            Assert.Equal(-6.0, ieee.G(1), precision: 12);
            Assert.Equal(-5.0, modulo.G(1), precision: 12);
        }

        [Fact]
        public void Ieee754Remainder_ComposedWithInnerExpressions_FollowsTheChainRule()
        {
            var (u, v) = DD2.Variables(0.6, 1.4);

            DD2 a = 5.0 + 1.1 * u - 0.4 * v + 0.3 * u * v;
            DD2 b = 1.3 + 0.2 * u - 0.1 * v;

            DD2 f = DD2.Ieee754Remainder(a, b);

            // Away from a break point the quotient is locally constant, so the remainder equals the
            // affine expression a - q*b and must agree with it in every derivative.
            double q = Math.Round(a.Value / b.Value, MidpointRounding.ToEven);
            DD2 affine = a - q * b;

            Assert.Equal(affine.Value, f.Value, precision: 12);
            Assert.Equal(affine.G(0), f.G(0), precision: 12);
            Assert.Equal(affine.G(1), f.G(1), precision: 12);
            Assert.Equal(affine.H(0, 0), f.H(0, 0), precision: 12);
            Assert.Equal(affine.H(0, 1), f.H(0, 1), precision: 12);
            Assert.Equal(affine.H(1, 1), f.H(1, 1), precision: 12);
        }

        [Fact]
        public void Ieee754Remainder_AgreesAcrossAllThreeModels()
        {
            double[] point = { 5.3, 1.3, 0.0 };
            const int size = 3, order = 2;

            var (sx, sy, _) = DD3.Variables(point[0], point[1], point[2]);
            DD3 statik = DD3.Ieee754Remainder(sx, sy);

            var d = DDScalar.Variables(point, order);
            DDScalar dynamik = Ieee754Remainder(d[0], d[1]);

            Span<double> destBuffer = stackalloc double[Kernel.GetDataLength(size, order)];
            var dest = new DDScalarSpan(destBuffer, size, order);
            new DDScalarSpan(d[0].AsSpan(), size, order)
                .Ieee754Remainder(new DDScalarSpan(d[1].AsSpan(), size, order), dest);

            AssertSameDerivatives("Ieee754Remainder", statik, dynamik, dest, size);
        }

        #endregion

        [Fact]
        public void MismatchedOperands_Throw()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 2);
            DDScalar b = DDScalar.Variable(0, 1.0, size: 3);

            Assert.Throws<InvalidOperationException>(() => Ieee754Remainder(a, b));
            Assert.Throws<InvalidOperationException>(() => FusedMultiplyAdd(a, a, b));
        }

        private static void AssertSameDerivatives(string what, in DD3 statik, in DDScalar dynamik, in DDScalarSpan span, int size)
        {
            Assert.Equal(statik.Value, dynamik.Value);
            Assert.Equal(statik.Value, span.Value);

            for (int i = 0; i < size; i++)
            {
                Assert.True(Math.Abs(statik.G(i) - dynamik.G(i)) < 1e-12, $"{what} dynamic G({i})");
                Assert.True(Math.Abs(statik.G(i) - span.G(i)) < 1e-12, $"{what} span G({i})");

                for (int j = 0; j < size; j++)
                {
                    Assert.True(Math.Abs(statik.H(i, j) - dynamik.H(i, j)) < 1e-12, $"{what} dynamic H({i},{j})");
                    Assert.True(Math.Abs(statik.H(i, j) - span.H(i, j)) < 1e-12, $"{what} span H({i},{j})");
                }
            }
        }
    }
}
