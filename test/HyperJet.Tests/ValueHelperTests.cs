using System;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Selection, sign, rounding, classification and the IEEE bit operations on the dynamic models.
    /// </summary>
    /// <remarks>
    /// These split into three derivative behaviours, and getting a member into the wrong group is the
    /// realistic mistake: selection hands back an operand and must carry its derivatives, rounding
    /// and <c>Sign</c> are piecewise constant and must drop them, and <c>BitIncrement</c>/<c>ScaleB</c>
    /// are shifts and scalings that must keep or scale them.
    /// </remarks>
    public class ValueHelperTests
    {
        private const int Size = 2, Order = 2;

        /// <summary>Two scalars with distinct, non-trivial derivatives so a mix-up is visible.</summary>
        private static (DDScalar A, DDScalar B) Operands()
        {
            var v = DDScalar.Variables(new[] { 0.8, 1.6 }, Order);
            return (Sin(v[0]) * v[1], Exp(v[1]) - 2.0 * v[0] * v[0]);
        }

        private static DDScalarSpan Wrap(in DDScalar a) => new DDScalarSpan(a.AsSpan(), Size, Order);

        private static void AssertSame(string what, in DDScalar expected, in DDScalar actual)
        {
            Assert.True(expected.Value.Equals(actual.Value), $"{what}: value {expected.Value:R} vs {actual.Value:R}");
            for (int i = 0; i < Size; i++)
            {
                Assert.True(expected.G(i).Equals(actual.G(i)), $"{what}: G({i})");
                for (int j = 0; j < Size; j++)
                    Assert.True(expected.H(i, j).Equals(actual.H(i, j)), $"{what}: H({i},{j})");
            }
        }

        private static void AssertSame(string what, in DDScalar expected, in DDScalarSpan actual)
        {
            Assert.True(expected.Value.Equals(actual.Value), $"{what}: value {expected.Value:R} vs {actual.Value:R}");
            for (int i = 0; i < Size; i++)
            {
                Assert.True(expected.G(i).Equals(actual.G(i)), $"{what}: G({i})");
                for (int j = 0; j < Size; j++)
                    Assert.True(expected.H(i, j).Equals(actual.H(i, j)), $"{what}: H({i},{j})");
            }
        }

        private static void AssertDerivativeFree(string what, in DDScalar f, double expectedValue)
        {
            Assert.Equal(expectedValue, f.Value);
            for (int i = 0; i < Size; i++)
            {
                Assert.Equal(0.0, f.G(i));
                for (int j = 0; j < Size; j++) Assert.Equal(0.0, f.H(i, j));
            }
        }

        #region Selection keeps the winner's derivatives

        [Fact]
        public void MinAndMax_ReturnTheSelectedOperandWithItsDerivatives()
        {
            var (a, b) = Operands();

            Assert.True(a.Value < b.Value, "the fixture should have a < b");

            AssertSame("Min", a, Min(a, b));
            AssertSame("Min (swapped)", a, Min(b, a));
            AssertSame("Max", b, Max(a, b));
            AssertSame("Max (swapped)", b, Max(b, a));
        }

        [Fact]
        public void MinMagnitudeAndMaxMagnitude_CompareAbsoluteValues()
        {
            var (a, b) = Operands();
            DDScalar negated = -b;

            Assert.True(Math.Abs(a.Value) < Math.Abs(negated.Value));

            AssertSame("MinMagnitude", a, MinMagnitude(a, negated));
            AssertSame("MaxMagnitude", negated, MaxMagnitude(a, negated));
        }

        [Fact]
        public void TheNumberVariants_LetANumberWinAgainstNaN()
        {
            var (a, _) = Operands();
            DDScalar nan = DDScalar.Constant(double.NaN, Size, Order);

            AssertSame("MinNumber", a, MinNumber(a, nan));
            AssertSame("MinNumber (swapped)", a, MinNumber(nan, a));
            AssertSame("MaxNumber", a, MaxNumber(nan, a));
            AssertSame("MinMagnitudeNumber", a, MinMagnitudeNumber(nan, a));
            AssertSame("MaxMagnitudeNumber", a, MaxMagnitudeNumber(nan, a));

            // Without the NaN guard the plain variants propagate it.
            Assert.True(double.IsNaN(Min(a, nan).Value) || Min(a, nan).Value == a.Value);
        }

        [Fact]
        public void Clamp_SelectsValueOrBound()
        {
            var v = DDScalar.Variables(new[] { 0.8, 1.6 }, Order);
            DDScalar low = Sin(v[0]);                       // ~0.717
            DDScalar high = Exp(v[1]) - 2.0;                // ~2.95
            DDScalar inside = v[0] + v[1];                  // 2.4

            AssertSame("Clamp inside", inside, Clamp(inside, low, high));
            AssertSame("Clamp below", low, Clamp(low - 1.0, low, high));
            AssertSame("Clamp above", high, Clamp(high + 1.0, low, high));

            Assert.Throws<ArgumentException>(() => Clamp(inside, high, low));
        }

        /// <summary>Returning an operand must not hand back a struct that shares its buffer.</summary>
        [Fact]
        public void SelectionReturnsAnIndependentScalar()
        {
            var (a, b) = Operands();

            DDScalar picked = Min(a, b);
            picked.Value = -999.0;

            Assert.NotEqual(-999.0, a.Value);
        }

        #endregion

        #region Piecewise-constant members drop the derivatives

        [Fact]
        public void SignAndRounding_ReturnConstants()
        {
            var v = DDScalar.Variables(new[] { 0.8, 1.6 }, Order);
            DDScalar f = 2.7 * v[0] * v[1];  // 3.456

            AssertDerivativeFree("Sign", Sign(f), 1.0);
            AssertDerivativeFree("Sign (negative)", Sign(-f), -1.0);
            AssertDerivativeFree("Round", Round(f), Math.Round(f.Value));
            AssertDerivativeFree("Round (digits)", Round(f, 2, MidpointRounding.ToEven), Math.Round(f.Value, 2, MidpointRounding.ToEven));
            AssertDerivativeFree("Round (mode)", Round(f, MidpointRounding.ToZero), Math.Round(f.Value, MidpointRounding.ToZero));
            AssertDerivativeFree("Floor", Floor(f), Math.Floor(f.Value));
            AssertDerivativeFree("Ceiling", Ceiling(f), Math.Ceiling(f.Value));
            AssertDerivativeFree("Truncate", Truncate(f), Math.Truncate(f.Value));
        }

        [Fact]
        public void CopySign_TakesTheMagnitudeAndTheOtherSign()
        {
            var (a, b) = Operands();

            AssertSame("CopySign positive", Abs(a), CopySign(a, b));
            AssertSame("CopySign negative", -Abs(a), CopySign(a, -b));
        }

        #endregion

        #region Shifts and scalings keep or scale the derivatives

        [Fact]
        public void BitIncrementAndDecrement_MoveOnlyTheValue()
        {
            var (a, _) = Operands();

            DDScalar up = BitIncrement(a);
            DDScalar down = BitDecrement(a);

            Assert.Equal(double.BitIncrement(a.Value), up.Value);
            Assert.Equal(double.BitDecrement(a.Value), down.Value);

            for (int i = 0; i < Size; i++)
            {
                Assert.Equal(a.G(i), up.G(i));
                Assert.Equal(a.G(i), down.G(i));
                for (int j = 0; j < Size; j++)
                {
                    Assert.Equal(a.H(i, j), up.H(i, j));
                    Assert.Equal(a.H(i, j), down.H(i, j));
                }
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(4)]
        [InlineData(-7)]
        public void ScaleB_ScalesEveryCoefficientExactly(int n)
        {
            var (a, _) = Operands();
            double factor = Math.ScaleB(1.0, n);

            DDScalar scaled = ScaleB(a, n);

            Assert.Equal(a.Value * factor, scaled.Value);
            for (int i = 0; i < Size; i++)
            {
                Assert.Equal(a.G(i) * factor, scaled.G(i));
                for (int j = 0; j < Size; j++) Assert.Equal(a.H(i, j) * factor, scaled.H(i, j));
            }
        }

        [Fact]
        public void ILogB_ReadsTheExponentOfTheValue()
        {
            var (a, _) = Operands();

            Assert.Equal(double.ILogB(a.Value), ILogB(a));
            Assert.Equal(double.ILogB(a.Value), ILogB(Wrap(a)));
        }

        #endregion

        #region Classification

        [Fact]
        public void Predicates_AgreeWithTheUnderlyingDouble()
        {
            foreach (double value in new[] { 0.0, -0.0, 1.5, -2.0, 4.0, 5.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.Epsilon })
            {
                DDScalar s = DDScalar.Constant(value, Size, Order);
                DDScalarSpan span = Wrap(s);

                Assert.Equal(double.IsFinite(value), IsFinite(s));
                Assert.Equal(double.IsFinite(value), IsFinite(span));
                Assert.Equal(double.IsNaN(value), IsNaN(s));
                Assert.Equal(double.IsNaN(value), IsNaN(span));
                Assert.Equal(double.IsInfinity(value), IsInfinity(s));
                Assert.Equal(double.IsPositiveInfinity(value), IsPositiveInfinity(s));
                Assert.Equal(double.IsNegativeInfinity(value), IsNegativeInfinity(s));
                Assert.Equal(double.IsNegative(value), IsNegative(s));
                Assert.Equal(double.IsPositive(value), IsPositive(s));
                Assert.Equal(double.IsInteger(value), IsInteger(s));
                Assert.Equal(double.IsEvenInteger(value), IsEvenInteger(s));
                Assert.Equal(double.IsOddInteger(value), IsOddInteger(s));
                Assert.Equal(double.IsNormal(value), IsNormal(s));
                Assert.Equal(double.IsSubnormal(value), IsSubnormal(s));
                Assert.Equal(double.IsRealNumber(value), IsRealNumber(s));

                Assert.Equal(value == 0.0, IsZero(s));
                Assert.False(IsComplexNumber(s));
                Assert.False(IsImaginaryNumber(s));
                Assert.True(IsCanonical(s));
            }
        }

        #endregion

        #region The three models agree

        [Fact]
        public void DDScalarSpan_MatchesTheDynamicModel()
        {
            var (a, b) = Operands();

            int n = Kernel.GetDataLength(Size, Order);
            Span<double> buffer = stackalloc double[n];
            var dest = new DDScalarSpan(buffer, Size, Order);

            Wrap(a).Min(Wrap(b), dest); AssertSame("span Min", Min(a, b), dest);
            Wrap(a).Max(Wrap(b), dest); AssertSame("span Max", Max(a, b), dest);
            Wrap(a).MinMagnitude(Wrap(b), dest); AssertSame("span MinMagnitude", MinMagnitude(a, b), dest);
            Wrap(a).MaxMagnitude(Wrap(b), dest); AssertSame("span MaxMagnitude", MaxMagnitude(a, b), dest);
            Wrap(a).MinNumber(Wrap(b), dest); AssertSame("span MinNumber", MinNumber(a, b), dest);
            Wrap(a).MaxNumber(Wrap(b), dest); AssertSame("span MaxNumber", MaxNumber(a, b), dest);
            Wrap(a).MinMagnitudeNumber(Wrap(b), dest); AssertSame("span MinMagnitudeNumber", MinMagnitudeNumber(a, b), dest);
            Wrap(a).MaxMagnitudeNumber(Wrap(b), dest); AssertSame("span MaxMagnitudeNumber", MaxMagnitudeNumber(a, b), dest);
            Wrap(a).Sign(dest); AssertSame("span Sign", Sign(a), dest);
            Wrap(a).CopySign(Wrap(-b), dest); AssertSame("span CopySign", CopySign(a, -b), dest);
            Wrap(a).Round(dest); AssertSame("span Round", Round(a), dest);
            Wrap(a).Round(2, MidpointRounding.ToEven, dest); AssertSame("span Round(digits)", Round(a, 2, MidpointRounding.ToEven), dest);
            Wrap(a).Round(MidpointRounding.ToZero, dest); AssertSame("span Round(mode)", Round(a, MidpointRounding.ToZero), dest);
            Wrap(a).Floor(dest); AssertSame("span Floor", Floor(a), dest);
            Wrap(a).Ceiling(dest); AssertSame("span Ceiling", Ceiling(a), dest);
            Wrap(a).Truncate(dest); AssertSame("span Truncate", Truncate(a), dest);
            Wrap(a).BitIncrement(dest); AssertSame("span BitIncrement", BitIncrement(a), dest);
            Wrap(a).BitDecrement(dest); AssertSame("span BitDecrement", BitDecrement(a), dest);
            Wrap(a).ScaleB(5, dest); AssertSame("span ScaleB", ScaleB(a, 5), dest);
        }

        [Fact]
        public void DDScalarSpan_Clamp_MatchesTheDynamicModel()
        {
            var v = DDScalar.Variables(new[] { 0.8, 1.6 }, Order);
            DDScalar low = Sin(v[0]);
            DDScalar high = Exp(v[1]) - 2.0;
            DDScalar inside = v[0] + v[1];

            int n = Kernel.GetDataLength(Size, Order);
            Span<double> buffer = stackalloc double[n];
            var dest = new DDScalarSpan(buffer, Size, Order);

            Wrap(inside).Clamp(Wrap(low), Wrap(high), dest);
            AssertSame("span Clamp", Clamp(inside, low, high), dest);
        }

        [Fact]
        public void GeneratedStruct_MatchesTheDynamicModel()
        {
            var (sx, sy) = DD2.Variables(0.8, 1.6);
            DD2 sa = DD2.Sin(sx) * sy;
            DD2 sb = DD2.Exp(sy) - 2.0 * sx * sx;

            var (a, b) = Operands();

            AssertSameAsStruct("Min", DD2.Min(sa, sb), Min(a, b));
            AssertSameAsStruct("Max", DD2.Max(sa, sb), Max(a, b));
            AssertSameAsStruct("MinMagnitude", DD2.MinMagnitude(sa, sb), MinMagnitude(a, b));
            AssertSameAsStruct("Sign", DD2.Sign(sa), Sign(a));
            AssertSameAsStruct("CopySign", DD2.CopySign(sa, -sb), CopySign(a, -b));
            AssertSameAsStruct("Round", DD2.Round(sa), Round(a));
            AssertSameAsStruct("Floor", DD2.Floor(sa), Floor(a));
            AssertSameAsStruct("BitIncrement", DD2.BitIncrement(sa), BitIncrement(a));
            AssertSameAsStruct("ScaleB", DD2.ScaleB(sa, 5), ScaleB(a, 5));
            Assert.Equal(DD2.ILogB(sa), ILogB(a));
        }

        private static void AssertSameAsStruct(string what, in DD2 expected, in DDScalar actual)
        {
            Assert.True(expected.Value.Equals(actual.Value), $"{what}: value {expected.Value:R} vs {actual.Value:R}");
            for (int i = 0; i < Size; i++)
            {
                Assert.True(expected.G(i).Equals(actual.G(i)), $"{what}: G({i})");
                for (int j = 0; j < Size; j++)
                    Assert.True(expected.H(i, j).Equals(actual.H(i, j)), $"{what}: H({i},{j})");
            }
        }

        #endregion
    }
}
