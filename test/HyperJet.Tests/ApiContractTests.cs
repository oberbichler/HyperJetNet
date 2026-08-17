using System;
using System.Globalization;
using System.Numerics;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;

namespace HyperJet.Tests
{
    /// <summary>
    /// Argument validation, generic-math surface, and the behaviours a caller can trip over.
    /// </summary>
    public class ApiContractTests
    {
        #region Construction and argument validation

        [Theory]
        [InlineData(-1, 2)]
        [InlineData(-5, 1)]
        public void DDScalar_NegativeSize_Throws(int size, int order)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new DDScalar(size, order));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(3)]
        [InlineData(-1)]
        public void DDScalar_UnsupportedOrder_Throws(int order)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new DDScalar(2, order));
        }

        [Fact]
        public void DDScalar_SizeZero_IsAPlainConstant()
        {
            var c = DDScalar.Constant(3.5, size: 0);

            Assert.Equal(0, c.Size);
            Assert.Equal(1, c.DataLength);
            Assert.Equal(3.5, c.Value);
            Assert.Empty(c.GetGradient());

            DDScalar sum = c + DDScalar.Constant(1.5, size: 0);
            Assert.Equal(5.0, sum.Value);
        }

        [Fact]
        public void DDScalar_GradientIndexOutOfRange_Throws()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 2);

            Assert.Throws<ArgumentOutOfRangeException>(() => a.G(-1));
            Assert.Throws<ArgumentOutOfRangeException>(() => a.G(2));
            Assert.Throws<ArgumentOutOfRangeException>(() => a.H(0, 2));
            Assert.Throws<ArgumentOutOfRangeException>(() => a.H(2, 0));

            // Seeding validates too, now that it is the only way in.
            Assert.Throws<ArgumentOutOfRangeException>(() => DDScalar.Variable(2, 1.0, size: 2));
            Assert.Throws<ArgumentOutOfRangeException>(() => DDScalar.Variable(-1, 1.0, size: 2));
        }

        [Fact]
        public void DDScalar_FirstOrder_HasNoHessian()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 2, order: 1);

            Assert.Throws<InvalidOperationException>(() => a.H(0, 0));
            Assert.Throws<InvalidOperationException>(() => a.GetHessian());
        }

        [Fact]
        public void DDScalar_Uninitialized_ThrowsOnAccess()
        {
            DDScalar a = default;

            Assert.Equal(0, a.DataLength);
            Assert.Throws<InvalidOperationException>(() => a.G(0));
            Assert.Throws<InvalidOperationException>(() => a.GetGradient());
            Assert.Equal(0.0, a.Value);
            Assert.True(a.AsSpan().IsEmpty);
            Assert.Equal("Uninitialized DDScalar", a.ToString());
        }

        [Fact]
        public void DDScalar_GetGradientIntoTooSmallSpan_Throws()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 3);
            double[] destination = new double[2];

            Assert.Throws<ArgumentException>(() => a.GetGradient(destination));
        }

        [Fact]
        public void DDScalarSpan_BufferTooSmall_Throws()
        {
            double[] buffer = new double[Kernel.GetDataLength(3, 2) - 1];

            Assert.Throws<ArgumentException>(() => new DDScalarSpan(buffer, 3, 2));
        }

        [Fact]
        public void DDScalarSpan_OversizedBuffer_IsSlicedToTheExactLength()
        {
            int n = Kernel.GetDataLength(2, 2);
            Span<double> buffer = stackalloc double[n + 7];

            var s = new DDScalarSpan(buffer, 2, 2);

            Assert.Equal(n, s.DataLength);
        }

        [Fact]
        public void DDScalarSpan_MismatchedOperands_Throw()
        {
            Span<double> a = stackalloc double[Kernel.GetDataLength(2, 2)];
            Span<double> b = stackalloc double[Kernel.GetDataLength(3, 2)];
            Span<double> d = stackalloc double[Kernel.GetDataLength(2, 2)];

            var x = new DDScalarSpan(a, 2, 2);
            var y = new DDScalarSpan(b, 3, 2);
            var dest = new DDScalarSpan(d, 2, 2);

            // Record.Exception takes a lambda, which cannot capture ref structs.
            try
            {
                x.Add(y, dest);
                Assert.Fail("Adding operands of different size should throw.");
            }
            catch (InvalidOperationException)
            {
                // expected
            }
        }

        [Fact]
        public void HyperJetMath_MismatchedOperands_Throw()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 2);
            DDScalar b = DDScalar.Variable(0, 1.0, size: 3);

            Assert.Throws<InvalidOperationException>(() => HyperJetMath.Atan2(a, b));
            Assert.Throws<InvalidOperationException>(() => HyperJetMath.Hypot(a, b));
        }

        #endregion

        #region Hazards worth knowing about

        /// <summary>
        /// <see cref="DDScalar"/> is a struct wrapping a shared <c>double[]</c>, so copies still
        /// alias — but <see cref="DDScalar.AsSpan"/> is now the only way to reach the buffer, and
        /// asking for the buffer is an explicit act. The accessors that made the sharing look like
        /// ordinary local assignment are gone.
        /// </summary>
        [Fact]
        public void DDScalar_CopiesShareTheirBufferOnlyThroughAsSpan()
        {
            DDScalar a = DDScalar.Variable(0, 1.0, size: 2);
            DDScalar copy = a;

            copy.AsSpan()[0] = 99.0;

            Assert.Equal(99.0, a.Value);
        }

        /// <summary>
        /// A guard against the mutating members coming back: they are what turned the shared buffer
        /// into a trap, because an assignment through a copy read as local.
        /// </summary>
        [Fact]
        public void DDScalar_ExposesNoPostConstructionMutation()
        {
            Assert.Null(typeof(DDScalar).GetMethod("SetG"));
            Assert.Null(typeof(DDScalar).GetMethod("SetH"));
            Assert.Null(typeof(DDScalar).GetProperty("Value")!.SetMethod);
        }

        /// <summary>
        /// The generated static structs are genuine value types — copying them is a deep copy,
        /// because the coefficients live in an inline array inside the struct.
        /// </summary>
        [Fact]
        public void DDScalarN_CopiesAreIndependent()
        {
            DD2 a = DD2.Variable(0, 1.0);
            DD2 copy = a;

            copy.Value = 99.0;
            copy.SetG(1, 7.0);

            Assert.Equal(1.0, a.Value);
            Assert.Equal(0.0, a.G(1));
        }

        /// <summary>
        /// The kernels read the operands while writing the destination, so a destination that
        /// aliases an operand would corrupt the derivatives. Overlap is rejected up front rather
        /// than silently producing wrong numbers.
        /// </summary>
        [Fact]
        public void DDScalarSpan_DestinationOverlappingAnOperand_Throws()
        {
            int n = Kernel.GetDataLength(2, 2);

            Span<double> aBuffer = stackalloc double[n];
            Span<double> bBuffer = stackalloc double[n];

            var a = DDScalarSpan.Variable(aBuffer, 0, 3.0, 2, 2);
            var b = DDScalarSpan.Variable(bBuffer, 1, 6.0, 2, 2);

            // Destination is the first operand.
            try
            {
                a.Multiply(b, a);
                Assert.Fail("Writing into an operand should throw.");
            }
            catch (ArgumentException)
            {
            }

            // Destination is the second operand.
            try
            {
                a.Multiply(b, b);
                Assert.Fail("Writing into an operand should throw.");
            }
            catch (ArgumentException)
            {
            }

            // Unary operations are covered too.
            try
            {
                a.Sin(a);
                Assert.Fail("Writing into an operand should throw.");
            }
            catch (ArgumentException)
            {
            }
        }

        /// <summary>
        /// Partial overlap — separate <see cref="DDScalarSpan"/> instances carved out of one buffer
        /// at wrong offsets — is caught as well, not just the exact-same-span case.
        /// </summary>
        [Fact]
        public void DDScalarSpan_PartiallyOverlappingDestination_Throws()
        {
            int n = Kernel.GetDataLength(2, 2);
            Span<double> pool = stackalloc double[2 * n];

            var a = DDScalarSpan.Variable(pool[..n], 0, 3.0, 2, 2);
            var shifted = new DDScalarSpan(pool.Slice(n - 1, n), 2, 2);

            try
            {
                a.Sin(shifted);
                Assert.Fail("A partially overlapping destination should throw.");
            }
            catch (ArgumentException)
            {
            }
        }

        [Fact]
        public void DDScalarSpan_DisjointDestination_IsAccepted()
        {
            int n = Kernel.GetDataLength(2, 2);
            Span<double> pool = stackalloc double[2 * n];

            var a = DDScalarSpan.Variable(pool[..n], 0, 3.0, 2, 2);
            var dest = new DDScalarSpan(pool[n..], 2, 2);

            a.Sin(dest);

            Assert.Equal(Math.Sin(3.0), dest.Value, precision: 12);
        }

        #endregion

        #region Generic math surface

        [Fact]
        public void Comparisons_UseTheScalarValueOnly()
        {
            var (x, y) = DD2.Variables(2.0, 3.0);

            Assert.True(x < y);
            Assert.True(y > x);
            DD2 sameAsX = DD2.Variable(0, 2.0);
            Assert.True(x <= sameAsX);
            Assert.True(x >= sameAsX);
            Assert.True(x != y);

            // Same value, different derivatives — still "equal" under the value-based comparison.
            Assert.True(DD2.Variable(0, 5.0) == DD2.Variable(1, 5.0));
            Assert.Equal(DD2.Variable(0, 5.0).GetHashCode(), DD2.Variable(1, 5.0).GetHashCode());
            Assert.Equal(0, DD2.Variable(0, 5.0).CompareTo(DD2.Variable(1, 5.0)));
        }

        [Fact]
        public void Constants_MatchTheUnderlyingType()
        {
            Assert.Equal(Math.PI, DD2.Pi.Value);
            Assert.Equal(Math.E, DD2.E.Value);
            Assert.Equal(Math.Tau, DD2.Tau.Value);
            Assert.Equal(1.0, DD2.One.Value);
            Assert.Equal(0.0, DD2.Zero.Value);
            Assert.Equal(-1.0, DD2.NegativeOne.Value);
            Assert.Equal(0.0, DD2.AdditiveIdentity.Value);
            Assert.Equal(1.0, DD2.MultiplicativeIdentity.Value);
            Assert.Equal(2, DD2.Radix);

            // Constants carry no derivative information.
            Assert.Equal(0.0, DD2.Pi.G(0));
        }

        [Fact]
        public void IncrementAndDecrement_ShiftTheValueAndKeepDerivatives()
        {
            DD2 x = DD2.Variable(0, 5.0);

            DD2 up = ++x;
            Assert.Equal(6.0, up.Value);
            Assert.Equal(1.0, up.G(0));

            DD2 y = DD2.Variable(0, 5.0);
            DD2 down = --y;
            Assert.Equal(4.0, down.Value);
            Assert.Equal(1.0, down.G(0));

            Assert.Equal(5.0, (+DD2.Variable(0, 5.0)).Value);
        }

        [Fact]
        public void MinMaxAndClamp_SelectAnOperandWithItsDerivatives()
        {
            var (x, y) = DD2.Variables(2.0, 3.0);

            Assert.Equal(3.0, DD2.Max(x, y).Value);
            Assert.Equal(1.0, DD2.Max(x, y).G(1));
            Assert.Equal(2.0, DD2.Min(x, y).Value);
            Assert.Equal(1.0, DD2.Min(x, y).G(0));

            Assert.Equal(2.5, DD2.Clamp(DD2.Constant(2.5), x, y).Value);
            Assert.Equal(3.0, DD2.Clamp(DD2.Constant(9.0), x, y).Value);
            Assert.Equal(2.0, DD2.Clamp(DD2.Constant(-9.0), x, y).Value);

            Assert.Equal(-3.0, DD2.MaxMagnitude(x, DD2.Constant(-3.0)).Value);
            Assert.Equal(2.0, DD2.MinMagnitude(x, DD2.Constant(-3.0)).Value);
        }

        [Fact]
        public void RoundingFunctions_ReturnConstants()
        {
            DD2 x = DD2.Variable(0, 2.7);

            Assert.Equal(3.0, DD2.Round(x).Value);
            Assert.Equal(2.0, DD2.Floor(x).Value);
            Assert.Equal(3.0, DD2.Ceiling(x).Value);
            Assert.Equal(2.0, DD2.Truncate(x).Value);
            Assert.Equal(2.7, DD2.Round(x, 1, MidpointRounding.ToEven).Value);
            Assert.Equal(1.0, DD2.Sign(x).Value);
            Assert.Equal(-1.0, DD2.Sign(DD2.Constant(-2.7)).Value);
            Assert.Equal(-2.7, DD2.CopySign(x, DD2.Constant(-1.0)).Value);

            // Rounding is piecewise constant, so the derivative is zero away from the break points.
            Assert.Equal(0.0, DD2.Round(x).G(0));
        }

        [Fact]
        public void ClassificationPredicates_DelegateToTheValue()
        {
            Assert.True(DD2.IsFinite(DD2.Constant(1.0)));
            Assert.True(DD2.IsNaN(DD2.Constant(double.NaN)));
            Assert.True(DD2.IsInfinity(DD2.Constant(double.PositiveInfinity)));
            Assert.True(DD2.IsNegativeInfinity(DD2.Constant(double.NegativeInfinity)));
            Assert.True(DD2.IsNegative(DD2.Constant(-1.0)));
            Assert.True(DD2.IsPositive(DD2.Constant(1.0)));
            Assert.True(DD2.IsZero(DD2.Constant(0.0)));
            Assert.True(DD2.IsInteger(DD2.Constant(4.0)));
            Assert.True(DD2.IsEvenInteger(DD2.Constant(4.0)));
            Assert.True(DD2.IsOddInteger(DD2.Constant(5.0)));
            Assert.True(DD2.IsRealNumber(DD2.Constant(1.0)));
            Assert.False(DD2.IsComplexNumber(DD2.Constant(1.0)));
            Assert.False(DD2.IsImaginaryNumber(DD2.Constant(1.0)));
            Assert.True(DD2.IsNormal(DD2.Constant(1.0)));
            Assert.False(DD2.IsSubnormal(DD2.Constant(1.0)));
        }

        [Fact]
        public void ParseAndFormat_RoundTripThroughTheValue()
        {
            DD2 parsed = DD2.Parse("2.5", CultureInfo.InvariantCulture);
            Assert.Equal(2.5, parsed.Value);
            Assert.Equal(0.0, parsed.G(0));

            Assert.True(DD2.TryParse("2.5", CultureInfo.InvariantCulture, out DD2 tryParsed));
            Assert.Equal(2.5, tryParsed.Value);

            Assert.True(DD2.TryParse("2.5".AsSpan(), CultureInfo.InvariantCulture, out DD2 fromSpan));
            Assert.Equal(2.5, fromSpan.Value);

            Assert.False(DD2.TryParse("not a number", CultureInfo.InvariantCulture, out _));

            Assert.Equal("2.50", parsed.ToString("F2", CultureInfo.InvariantCulture));

            Span<char> buffer = stackalloc char[16];
            Assert.True(parsed.TryFormat(buffer, out int written, "F2", CultureInfo.InvariantCulture));
            Assert.Equal("2.50", new string(buffer[..written]));
        }

        [Fact]
        public void ToString_ShowsValueGradientAndHessian()
        {
            var (x, y) = DD2.Variables(1.0, 2.0);
            string text = (x * y).ToString();

            Assert.Contains("g:", text);
            Assert.Contains("H:", text);
        }

        [Fact]
        public void FloatBacking_ProducesTheSameDerivatives()
        {
            var (xf, yf) = DDScalar2<float>.Variables(3.0f, 6.0f);
            DDScalar2<float> f = (xf * yf) / (xf - yf);

            var (xd, yd) = DD2.Variables(3.0, 6.0);
            DD2 d = (xd * yd) / (xd - yd);

            Assert.Equal((float)d.Value, f.Value, tolerance: 1e-5f);
            Assert.Equal((float)d.G(0), f.G(0), tolerance: 1e-5f);
            Assert.Equal((float)d.H(0, 0), f.H(0, 0), tolerance: 1e-5f);
        }

        #endregion

        #region Vector3D

        [Fact]
        public void Vector3D_NormalizeAndCross_CarryDerivatives()
        {
            var (x, y, z) = DDScalar3<double>.Variables(1.0, 2.0, 3.0);
            var u = new Vector3D<DDScalar3<double>>(x, y, z);

            DDScalar3<double> lengthSquared = u.LengthSquared();
            Assert.Equal(14.0, lengthSquared.Value, precision: 12);
            Assert.Equal(2.0, lengthSquared.G(0), precision: 12);  // d(x^2+y^2+z^2)/dx = 2x
            Assert.Equal(2.0, lengthSquared.H(0, 0), precision: 12);

            DDScalar3<double> length = u.Length();
            Assert.Equal(Math.Sqrt(14.0), length.Value, precision: 12);
            Assert.Equal(1.0 / Math.Sqrt(14.0), length.G(0), precision: 12); // d|u|/dx = x/|u|

            Vector3D<DDScalar3<double>> unit = u.Normalize();
            Assert.Equal(1.0, unit.LengthSquared().Value, precision: 12);
            // The normalized vector has unit length everywhere, so its length is stationary.
            Assert.Equal(0.0, unit.Length().G(0), precision: 10);

            // u x u == 0 identically, in value and in all derivatives.
            Vector3D<DDScalar3<double>> selfCross = u.Cross(u);
            Assert.Equal(0.0, selfCross.X.Value);
            Assert.Equal(0.0, selfCross.X.G(1), precision: 12);
            Assert.Equal(0.0, selfCross.Y.H(0, 2), precision: 12);

            // (u x v) . u == 0 identically.
            var v = new Vector3D<DDScalar3<double>>(
                DDScalar3<double>.Constant(4.0),
                DDScalar3<double>.Constant(5.0),
                DDScalar3<double>.Constant(6.0));
            DDScalar3<double> orthogonal = u.Cross(v).Dot(u);
            Assert.Equal(0.0, orthogonal.Value, precision: 12);
            Assert.Equal(0.0, orthogonal.G(0), precision: 12);
            Assert.Equal(0.0, orthogonal.H(0, 1), precision: 12);
        }

        [Fact]
        public void Vector3D_ZeroVector_NormalizesToItself()
        {
            var zero = new Vector3D<double>(0.0, 0.0, 0.0);
            Vector3D<double> normalized = zero.Normalize();

            Assert.Equal(0.0, normalized.X);
            Assert.Equal(0.0, normalized.Y);
            Assert.Equal(0.0, normalized.Z);
        }

        #endregion
    }
}
