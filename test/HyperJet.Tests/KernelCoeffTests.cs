using Xunit;
using HyperJet;

namespace HyperJet.Tests
{
    /// <summary>
    /// The kernels broadcast a coefficient's scalar factor into a SIMD vector for the bulk of the
    /// data and fall back to element-wise <see cref="ICoeff.Multiply"/> for the remainder. These
    /// tests pin down that both paths agree — including for coefficient types the kernel has no
    /// built-in knowledge of.
    /// </summary>
    public class KernelCoeffTests
    {
        /// <summary>A coefficient with a runtime factor that is none of the built-in tag types.</summary>
        private readonly struct ScaleCoeff : ICoeff
        {
            private readonly double _factor;

            public ScaleCoeff(double factor) => _factor = factor;

            public double Multiply(double val) => _factor * val;
        }

        // Sizes 1..20 give data lengths 3..231, crossing every vectorization threshold
        // (Vector128 at n>=3, Vector256 at n>=5, Vector512 at n>=9) and every remainder alignment.
        public static TheoryData<int> Sizes
        {
            get
            {
                var data = new TheoryData<int>();
                for (int size = 1; size <= 20; size++) data.Add(size);
                return data;
            }
        }

        private static double[] Ramp(int length, double offset)
        {
            double[] values = new double[length];
            for (int i = 0; i < length; i++) values[i] = offset + i;
            return values;
        }

        [Theory]
        [MemberData(nameof(Sizes))]
        public void Unary_CustomCoeff_ShouldApplyFactorAcrossVectorizedAndRemainderElements(int size)
        {
            const int order = 2;
            const double factor = 7.0;

            int n = Kernel.GetDataLength(size, order);
            double[] a = Ramp(n, 1.0);
            double[] r = new double[n];

            // daa = ZeroCoeff, so only the first-order propagation contributes.
            Kernel.Unary<FalseTag, ScaleCoeff, ZeroCoeff>(
                a, -1.5, new ScaleCoeff(factor), default, r, size, order);

            Assert.Equal(-1.5, r[0]);
            for (int i = 1; i < n; i++)
            {
                Assert.Equal(factor * a[i], r[i], precision: 12);
            }
        }

        [Theory]
        [MemberData(nameof(Sizes))]
        public void Binary_CustomCoeff_ShouldApplyBothFactorsAcrossAllElements(int size)
        {
            const int order = 2;
            const double factorA = 3.0;
            const double factorB = -5.0;

            int n = Kernel.GetDataLength(size, order);
            double[] a = Ramp(n, 1.0);
            double[] b = Ramp(n, 100.0);
            double[] r = new double[n];

            Kernel.Binary<FalseTag, ScaleCoeff, ScaleCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a, b, 2.5,
                new ScaleCoeff(factorA), new ScaleCoeff(factorB), default, default, default,
                r, size, order);

            Assert.Equal(2.5, r[0]);
            for (int i = 1; i < n; i++)
            {
                Assert.Equal(factorA * a[i] + factorB * b[i], r[i], precision: 12);
            }
        }

        [Theory]
        [MemberData(nameof(Sizes))]
        public void Ternary_CustomCoeff_ShouldApplyAllThreeFactorsAcrossAllElements(int size)
        {
            const int order = 2;
            const double factorA = 2.0;
            const double factorB = -0.5;
            const double factorC = 11.0;

            int n = Kernel.GetDataLength(size, order);
            double[] a = Ramp(n, 1.0);
            double[] b = Ramp(n, 100.0);
            double[] c = Ramp(n, 1000.0);
            double[] r = new double[n];

            Kernel.Ternary<FalseTag, ScaleCoeff, ScaleCoeff, ScaleCoeff,
                           ZeroCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a, b, c, 0.25,
                new ScaleCoeff(factorA), new ScaleCoeff(factorB), new ScaleCoeff(factorC),
                default, default, default, default, default, default,
                r, size, order);

            Assert.Equal(0.25, r[0]);
            for (int i = 1; i < n; i++)
            {
                Assert.Equal(factorA * a[i] + factorB * b[i] + factorC * c[i], r[i], precision: 12);
            }
        }

        /// <summary>
        /// A custom coefficient carrying the same factor as a built-in tag must produce bit-identical
        /// results, proving the vectorized path no longer special-cases known types.
        /// </summary>
        [Theory]
        [MemberData(nameof(Sizes))]
        public void CustomCoeff_ShouldMatchEquivalentBuiltInTag(int size)
        {
            const int order = 2;

            int n = Kernel.GetDataLength(size, order);
            double[] a = Ramp(n, 1.0);
            double[] b = Ramp(n, 100.0);

            double[] withTags = new double[n];
            Kernel.Binary<FalseTag, OneCoeff, MinusOneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a, b, 0.0, default, default, default, default, default, withTags, size, order);

            double[] withCustom = new double[n];
            Kernel.Binary<FalseTag, ScaleCoeff, ScaleCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a, b, 0.0,
                new ScaleCoeff(1.0), new ScaleCoeff(-1.0), default, default, default,
                withCustom, size, order);

            Assert.Equal(withTags, withCustom);
        }
    }
}
