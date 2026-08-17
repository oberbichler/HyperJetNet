using System;
using Xunit;

namespace HyperJet.Tests
{
    /// <summary>
    /// Holds the vectorized <see cref="Kernel"/> against a straightforward scalar reference
    /// implementation of the same recurrence.
    /// </summary>
    /// <remarks>
    /// Sizes 0..30 at order 2 give data lengths 1..496, which crosses every vectorization threshold
    /// (Vector128 at n&gt;=3, Vector256 at n&gt;=5, Vector512 at n&gt;=9) and hits every possible
    /// remainder length, so a mis-computed loop bound cannot slip through.
    /// </remarks>
    public class KernelTests
    {
        public static TheoryData<int, int> SizesAndOrders
        {
            get
            {
                var data = new TheoryData<int, int>();
                for (int size = 0; size <= 30; size++)
                {
                    data.Add(size, 1);
                    data.Add(size, 2);
                }
                return data;
            }
        }

        public static TheoryData<int> Sizes
        {
            get
            {
                var data = new TheoryData<int>();
                for (int size = 0; size <= 30; size++) data.Add(size);
                return data;
            }
        }

        private static double[] Sample(int length, int seed)
        {
            var rng = new Random(seed);
            double[] values = new double[length];
            for (int i = 0; i < length; i++) values[i] = rng.NextDouble() * 4.0 - 2.0;
            return values;
        }

        #region Scalar reference implementations

        private static double[] UnaryReference(double[] a, double f, double da, double daa, double[] seed, int size, int order, bool increment)
        {
            double[] r = (double[])seed.Clone();
            int n = Kernel.GetDataLength(size, order);

            r[0] = f;
            for (int i = 1; i < n; i++)
            {
                double term = da * a[i];
                if (increment) r[i] += term; else r[i] = term;
            }

            if (order < 2) return r;

            int k = 1 + size;
            for (int i = 0; i < size; i++)
            {
                double ca = daa * a[1 + i];
                for (int j = i; j < size; j++) r[k++] += ca * a[1 + j];
            }
            return r;
        }

        private static double[] BinaryReference(double[] a, double[] b, double f,
            double da, double db, double daa, double dab, double dbb,
            double[] seed, int size, int order, bool increment)
        {
            double[] r = (double[])seed.Clone();
            int n = Kernel.GetDataLength(size, order);

            r[0] = f;
            for (int i = 1; i < n; i++)
            {
                double term = da * a[i] + db * b[i];
                if (increment) r[i] += term; else r[i] = term;
            }

            if (order < 2) return r;

            int k = 1 + size;
            for (int i = 0; i < size; i++)
            {
                double ca = daa * a[1 + i] + dab * b[1 + i];
                double cb = dab * a[1 + i] + dbb * b[1 + i];
                for (int j = i; j < size; j++) r[k++] += ca * a[1 + j] + cb * b[1 + j];
            }
            return r;
        }

        private static double[] TernaryReference(double[] a, double[] b, double[] c, double f,
            double da, double db, double dc, double daa, double dab, double dac, double dbb, double dbc, double dcc,
            double[] seed, int size, int order, bool increment)
        {
            double[] r = (double[])seed.Clone();
            int n = Kernel.GetDataLength(size, order);

            r[0] = f;
            for (int i = 1; i < n; i++)
            {
                double term = da * a[i] + db * b[i] + dc * c[i];
                if (increment) r[i] += term; else r[i] = term;
            }

            if (order < 2) return r;

            int k = 1 + size;
            for (int i = 0; i < size; i++)
            {
                double ai = a[1 + i], bi = b[1 + i], ci = c[1 + i];
                double ca = daa * ai + dab * bi + dac * ci;
                double cb = dab * ai + dbb * bi + dbc * ci;
                double cc = dac * ai + dbc * bi + dcc * ci;
                for (int j = i; j < size; j++) r[k++] += ca * a[1 + j] + cb * b[1 + j] + cc * c[1 + j];
            }
            return r;
        }

        #endregion

        [Theory]
        [MemberData(nameof(SizesAndOrders))]
        public void Unary_ValueCoefficients_MatchScalarReference(int size, int order)
        {
            int n = Kernel.GetDataLength(size, order);
            double[] a = Sample(n, seed: 1000 + size);
            const double f = 1.25, da = -0.75, daa = 2.5;

            foreach (bool increment in new[] { false, true })
            {
                double[] seed = Sample(n, seed: 2000 + size);
                double[] expected = UnaryReference(a, f, da, daa, seed, size, order, increment);
                double[] actual = (double[])seed.Clone();

                if (increment)
                {
                    Kernel.Unary<TrueTag, ValueCoeff, ValueCoeff>(
                        a, f, new ValueCoeff(da), new ValueCoeff(daa), actual, size, order);
                }
                else
                {
                    Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                        a, f, new ValueCoeff(da), new ValueCoeff(daa), actual, size, order);
                }

                AssertClose(expected, actual, $"unary size={size} order={order} increment={increment}");
            }
        }

        [Theory]
        [MemberData(nameof(SizesAndOrders))]
        public void Binary_ValueCoefficients_MatchScalarReference(int size, int order)
        {
            int n = Kernel.GetDataLength(size, order);
            double[] a = Sample(n, seed: 3000 + size);
            double[] b = Sample(n, seed: 4000 + size);
            const double f = -0.5, da = 1.75, db = -2.25, daa = 0.5, dab = -1.5, dbb = 3.0;

            foreach (bool increment in new[] { false, true })
            {
                double[] seed = Sample(n, seed: 5000 + size);
                double[] expected = BinaryReference(a, b, f, da, db, daa, dab, dbb, seed, size, order, increment);
                double[] actual = (double[])seed.Clone();

                if (increment)
                {
                    Kernel.Binary<TrueTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                        a, b, f, new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                        actual, size, order);
                }
                else
                {
                    Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                        a, b, f, new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                        actual, size, order);
                }

                AssertClose(expected, actual, $"binary size={size} order={order} increment={increment}");
            }
        }

        [Theory]
        [MemberData(nameof(SizesAndOrders))]
        public void Ternary_ValueCoefficients_MatchScalarReference(int size, int order)
        {
            int n = Kernel.GetDataLength(size, order);
            double[] a = Sample(n, seed: 6000 + size);
            double[] b = Sample(n, seed: 7000 + size);
            double[] c = Sample(n, seed: 8000 + size);
            const double f = 3.5, da = 0.25, db = -1.25, dc = 2.0, daa = -0.5, dab = 1.5, dac = -2.5, dbb = 0.75, dbc = -0.25, dcc = 1.25;

            foreach (bool increment in new[] { false, true })
            {
                double[] seed = Sample(n, seed: 9000 + size);
                double[] expected = TernaryReference(a, b, c, f, da, db, dc, daa, dab, dac, dbb, dbc, dcc, seed, size, order, increment);
                double[] actual = (double[])seed.Clone();

                if (increment)
                {
                    Kernel.Ternary<TrueTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                        a, b, c, f,
                        new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(dc),
                        new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dac),
                        new ValueCoeff(dbb), new ValueCoeff(dbc), new ValueCoeff(dcc),
                        actual, size, order);
                }
                else
                {
                    Kernel.Ternary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                        a, b, c, f,
                        new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(dc),
                        new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dac),
                        new ValueCoeff(dbb), new ValueCoeff(dbc), new ValueCoeff(dcc),
                        actual, size, order);
                }

                AssertClose(expected, actual, $"ternary size={size} order={order} increment={increment}");
            }
        }

        /// <summary>
        /// The One/MinusOne tags must behave exactly like <c>ValueCoeff(±1)</c>, bit for bit —
        /// they only exist so the JIT can fold the multiplication away.
        /// </summary>
        [Theory]
        [MemberData(nameof(Sizes))]
        public void Binary_UnitTags_ProduceIdenticalResultsToValueCoefficients(int size)
        {
            const int order = 2;
            int n = Kernel.GetDataLength(size, order);
            double[] a = Sample(n, seed: 11000 + size);
            double[] b = Sample(n, seed: 12000 + size);

            double[] tagged = new double[n];
            Kernel.Binary<FalseTag, OneCoeff, MinusOneCoeff, ZeroCoeff, OneCoeff, ZeroCoeff>(
                a, b, 0.5, default, default, default, default, default, tagged, size, order);

            double[] valued = new double[n];
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                a, b, 0.5,
                new ValueCoeff(1.0), new ValueCoeff(-1.0), new ValueCoeff(0.0), new ValueCoeff(1.0), new ValueCoeff(0.0),
                valued, size, order);

            Assert.Equal(valued, tagged);
        }

        /// <summary>
        /// The generic <c>Kernel.Unary&lt;T, ...&gt;</c> overload is a separate, non-vectorized
        /// implementation. It must agree with the <c>double</c> overload.
        /// </summary>
        [Theory]
        [MemberData(nameof(SizesAndOrders))]
        public void GenericOverloads_AgreeWithDoubleOverloads(int size, int order)
        {
            int n = Kernel.GetDataLength(size, order);
            double[] a = Sample(n, seed: 13000 + size);
            double[] b = Sample(n, seed: 14000 + size);
            const double f = 0.125, da = 1.5, db = -0.5, daa = 2.0, dab = -1.0, dbb = 0.25;

            double[] fromDouble = new double[n];
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                a, b, f, new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                fromDouble, size, order);

            double[] fromGeneric = new double[n];
            Kernel.Binary<double, FalseTag, ValueCoeff<double>, ValueCoeff<double>, ValueCoeff<double>, ValueCoeff<double>, ValueCoeff<double>>(
                a, b, f,
                new ValueCoeff<double>(da), new ValueCoeff<double>(db), new ValueCoeff<double>(daa),
                new ValueCoeff<double>(dab), new ValueCoeff<double>(dbb),
                fromGeneric, size, order);

            AssertClose(fromDouble, fromGeneric, $"generic vs double size={size} order={order}");
        }

        [Theory]
        [MemberData(nameof(SizesAndOrders))]
        public void GetDataLength_And_GetSizeFromDataLength_RoundTrip(int size, int order)
        {
            int length = Kernel.GetDataLength(size, order);
            Assert.Equal(size, Kernel.GetSizeFromDataLength(length, order));
        }

        [Theory]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(5)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(9)]
        public void GetSizeFromDataLength_InvalidLength_Throws(int length)
        {
            Assert.Throws<ArgumentException>(() => Kernel.GetSizeFromDataLength(length, order: 2));
        }

        /// <summary>
        /// With a zero first-derivative coefficient but a non-zero second-derivative coefficient,
        /// the second-order chain-rule term <c>daa * grad(a) grad(a)^T</c> still contributes and
        /// must land in the Hessian block.
        /// </summary>
        [Fact]
        public void Unary_ZeroFirstDerivativeWithNonZeroSecondDerivative_StillPropagatesHessian()
        {
            const int size = 3, order = 2;
            int n = Kernel.GetDataLength(size, order);

            double[] a = new double[n];
            a[0] = 1.0;
            a[1] = 2.0; a[2] = 3.0; a[3] = 4.0; // gradient of the inner expression

            const double daa = 5.0;

            double[] actual = new double[n];
            Kernel.Unary<FalseTag, ZeroCoeff, ValueCoeff>(
                a, 0.0, default, new ValueCoeff(daa), actual, size, order);

            double[] expected = UnaryReference(a, 0.0, 0.0, daa, new double[n], size, order, increment: false);
            AssertClose(expected, actual, "zero da with non-zero daa");
        }

        /// <summary>
        /// In assign mode (<see cref="FalseTag"/>) the kernel owns every slot of the destination.
        /// A zero coefficient means "write zero", not "leave whatever was there".
        /// </summary>
        [Fact]
        public void Unary_AssignModeWithZeroCoefficient_ClearsDirtyDestination()
        {
            const int size = 3, order = 2;
            int n = Kernel.GetDataLength(size, order);

            double[] a = Sample(n, seed: 42);
            double[] actual = new double[n];
            Array.Fill(actual, 99.0); // reused buffer holding stale data

            Kernel.Unary<FalseTag, ZeroCoeff, ZeroCoeff>(a, 7.0, default, default, actual, size, order);

            double[] expected = new double[n];
            expected[0] = 7.0;
            AssertClose(expected, actual, "assign mode must not leak stale destination data");
        }

        private static void AssertClose(double[] expected, double[] actual, string what)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                double tolerance = 1e-12 * (1.0 + Math.Abs(expected[i]));
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                    $"{what}: index {i} expected {expected[i]:R}, got {actual[i]:R}");
            }
        }
    }
}
