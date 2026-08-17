using System;
using System.Numerics;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using Inner = HyperJet.DDScalar1<double>;
using Outer = HyperJet.DDScalar1<HyperJet.DDScalar1<double>>;

namespace HyperJet.Tests
{
    /// <summary>
    /// The generated structs implement <see cref="IFloatingPointIeee754{TSelf}"/>, not merely
    /// <see cref="IFloatingPoint{TSelf}"/>.
    /// </summary>
    /// <remarks>
    /// Two things follow. Generic code written against the usual constraint accepts dual numbers,
    /// and — because the structs constrain their own coefficient type to that same interface — the
    /// types can be nested, which raises the reachable derivative order from two to four.
    /// </remarks>
    public class FloatingPointIeee754Tests
    {
        #region The constraint is satisfied

        // Constrained to the IEEE-754 interface rather than IFloatingPoint. That this file compiles
        // is the assertion; the bodies only make the constraint load-bearing.
        private static T HalfWay<T>(T a, T b) where T : IFloatingPointIeee754<T> =>
            T.FusedMultiplyAdd(b - a, T.One / (T.One + T.One), a);

        private static bool LooksFinite<T>(T a) where T : IFloatingPointIeee754<T> =>
            T.IsFinite(a) && a != T.NaN && a != T.PositiveInfinity;

        [Fact]
        public void GenericCodeConstrainedToTheIeee754Interface_AcceptsDualNumbers()
        {
            var (x, y) = DD2.Variables(2.0, 6.0);

            DD2 mid = HalfWay(x, y);

            Assert.Equal(4.0, mid.Value, precision: 12);
            Assert.Equal(0.5, mid.G(0), precision: 12);
            Assert.Equal(0.5, mid.G(1), precision: 12);

            Assert.True(LooksFinite(mid));
        }

        #endregion

        #region Special values

        [Fact]
        public void SpecialValues_MirrorTheCoefficientTypeAndCarryNoDerivatives()
        {
            Assert.Equal(double.Epsilon, DD2.Epsilon.Value);
            Assert.Equal(double.NegativeZero, DD2.NegativeZero.Value);
            Assert.Equal(double.PositiveInfinity, DD2.PositiveInfinity.Value);
            Assert.Equal(double.NegativeInfinity, DD2.NegativeInfinity.Value);
            Assert.True(double.IsNaN(DD2.NaN.Value));

            Assert.Equal(0.0, DD2.Epsilon.G(0));
            Assert.Equal(0.0, DD2.PositiveInfinity.G(1));

            // The classification predicates agree with the special values.
            Assert.True(DD2.IsNaN(DD2.NaN));
            Assert.True(DD2.IsPositiveInfinity(DD2.PositiveInfinity));
            Assert.True(DD2.IsNegativeInfinity(DD2.NegativeInfinity));
            Assert.True(DD2.IsNegative(DD2.NegativeZero));
            Assert.True(DD2.IsSubnormal(DD2.Epsilon));
        }

        #endregion

        #region Bit stepping and scaling

        /// <summary>
        /// Within a binade the step to the neighbouring representable value is a constant, so the
        /// operation is <c>x + c</c> and the derivatives must survive it. This is what separates it
        /// from the piecewise-constant <c>Round</c> family, which correctly returns a constant.
        /// </summary>
        [Fact]
        public void BitIncrementAndDecrement_MoveTheValueAndKeepTheDerivatives()
        {
            var (x, y) = DD2.Variables(1.0, 2.0);
            DD2 f = x * y + DD2.Sin(x);

            DD2 up = DD2.BitIncrement(f);
            DD2 down = DD2.BitDecrement(f);

            Assert.Equal(double.BitIncrement(f.Value), up.Value);
            Assert.Equal(double.BitDecrement(f.Value), down.Value);

            foreach (DD2 stepped in new[] { up, down })
            {
                Assert.Equal(f.G(0), stepped.G(0));
                Assert.Equal(f.G(1), stepped.G(1));
                Assert.Equal(f.H(0, 0), stepped.H(0, 0));
                Assert.Equal(f.H(0, 1), stepped.H(0, 1));
                Assert.Equal(f.H(1, 1), stepped.H(1, 1));
            }

            // Contrast: Round is piecewise constant, so it does drop the derivatives.
            Assert.Equal(0.0, DD2.Round(f).G(0));
        }

        [Fact]
        public void ILogB_ReadsTheExponentOfTheValue()
        {
            Assert.Equal(double.ILogB(12.5), DD2.ILogB(DD2.Constant(12.5)));
            Assert.Equal(double.ILogB(0.001), DD2.ILogB(DD2.Constant(0.001)));
        }

        /// <summary>
        /// Scaling by a power of two is linear, so every coefficient scales with it — and because
        /// the scaling is exact in binary floating point, the comparison can be exact too.
        /// </summary>
        [Theory]
        [InlineData(0)]
        [InlineData(3)]
        [InlineData(-5)]
        [InlineData(40)]
        public void ScaleB_ScalesEveryCoefficientExactly(int n)
        {
            var (x, y) = DD2.Variables(1.3, 2.7);
            DD2 f = x * y + DD2.Sin(x) * y;

            DD2 scaled = DD2.ScaleB(f, n);
            double factor = Math.ScaleB(1.0, n);

            Assert.Equal(f.Value * factor, scaled.Value);
            Assert.Equal(f.G(0) * factor, scaled.G(0));
            Assert.Equal(f.G(1) * factor, scaled.G(1));
            Assert.Equal(f.H(0, 0) * factor, scaled.H(0, 0));
            Assert.Equal(f.H(0, 1) * factor, scaled.H(0, 1));
            Assert.Equal(f.H(1, 1) * factor, scaled.H(1, 1));
        }

        #endregion

        #region Nesting: fourth-order derivatives

        /// <summary>
        /// <c>DDScalar1&lt;DDScalar1&lt;double&gt;&gt;</c> only compiles because the inner type now
        /// satisfies the coefficient constraint. Each level contributes two orders, so four
        /// derivatives become reachable — and they are produced by the same second-order machinery
        /// applied to itself, which makes this an independent check of it.
        /// </summary>
        [Theory]
        [InlineData(1.3)]
        [InlineData(-0.6)]
        [InlineData(2.5)]
        public void NestedScalars_ProduceFourthOrderDerivativesOfAPolynomial(double x0)
        {
            Outer u = Seed(x0);

            Outer f = u * u * u * u * u; // x^5

            Assert.Equal(Math.Pow(x0, 5), f.Value.Value, precision: 10);
            Assert.Equal(5 * Math.Pow(x0, 4), f.G(0).Value, precision: 10);
            Assert.Equal(20 * Math.Pow(x0, 3), f.H(0, 0).Value, precision: 10);
            Assert.Equal(60 * Math.Pow(x0, 2), f.H(0, 0).G(0), precision: 10);
            Assert.Equal(120 * x0, f.H(0, 0).H(0, 0), precision: 10);
        }

        [Fact]
        public void NestedScalars_ProduceFourthOrderDerivativesOfTranscendentals()
        {
            const double x0 = 0.7;
            Outer u = Seed(x0);

            // d4/dx4 sin(x) = sin(x)
            Assert.Equal(Math.Sin(x0), Outer.Sin(u).H(0, 0).H(0, 0), precision: 10);

            // d4/dx4 cos(x) = cos(x)
            Assert.Equal(Math.Cos(x0), Outer.Cos(u).H(0, 0).H(0, 0), precision: 10);

            // d4/dx4 exp(x) = exp(x)
            Assert.Equal(Math.Exp(x0), Outer.Exp(u).H(0, 0).H(0, 0), precision: 10);

            // d4/dx4 log(x) = -6/x^4
            Assert.Equal(-6.0 / Math.Pow(x0, 4), Outer.Log(u).H(0, 0).H(0, 0), precision: 10);

            // d4/dx4 sqrt(x) = -15/16 * x^(-7/2)
            Assert.Equal(-15.0 / 16.0 * Math.Pow(x0, -3.5), Outer.Sqrt(u).H(0, 0).H(0, 0), precision: 10);
        }

        /// <summary>Seeds the same variable at both levels, so all four orders refer to one x.</summary>
        private static Outer Seed(double x0) => Outer.Variable(0, Inner.Variable(0, x0));

        #endregion
    }
}
