using System;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using DD3 = HyperJet.DDScalar3<double>;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// <c>Evaluate</c> is the local Taylor model: <c>f(x + d) = f(x) + grad(f)·d + ½·dᵀHd</c>.
    /// </summary>
    /// <remarks>
    /// The central test is exactness on a quadratic. A second-order expansion of a quadratic
    /// reproduces the function everywhere, for arbitrarily large offsets, so comparing against the
    /// closed-form polynomial checks the gradient, the Hessian, the triangular packing and the
    /// factor one half all at once — and would fail for any of them being wrong.
    /// </remarks>
    public class EvaluateTests
    {
        private const double Constant = 0.37;

        private static double Linear(int i) => 0.5 + 0.2 * i;

        /// <summary>Symmetric by construction, so it is a valid Hessian.</summary>
        private static double Curvature(int i, int j) => 0.3 + 0.1 * (i + j);

        /// <summary>f(x) = c + b·x + ½·xᵀAx, evaluated in plain arithmetic.</summary>
        private static double Quadratic(ReadOnlySpan<double> x)
        {
            double result = Constant;

            for (int i = 0; i < x.Length; i++) result += Linear(i) * x[i];

            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < x.Length; j++)
                    result += 0.5 * Curvature(i, j) * x[i] * x[j];

            return result;
        }

        /// <summary>The same polynomial, built through the dynamic AD model.</summary>
        private static DDScalar Quadratic(DDScalar[] x)
        {
            int n = x.Length;
            DDScalar result = DDScalar.Constant(Constant, n, x[0].Order);

            for (int i = 0; i < n; i++) result += Linear(i) * x[i];

            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    result += (0.5 * Curvature(i, j)) * x[i] * x[j];

            return result;
        }

        private static double[] Point(int n)
        {
            double[] x = new double[n];
            for (int i = 0; i < n; i++) x[i] = 0.4 - 0.13 * i + 0.02 * i * i;
            return x;
        }

        private static double[] Offset(int n, double scale)
        {
            double[] d = new double[n];
            for (int i = 0; i < n; i++) d[i] = scale * (0.7 - 0.31 * i);
            return d;
        }

        public static TheoryData<int> Sizes
        {
            get
            {
                var data = new TheoryData<int>();
                for (int n = 1; n <= 10; n++) data.Add(n);
                return data;
            }
        }

        [Theory]
        [MemberData(nameof(Sizes))]
        public void OnAQuadratic_TheExpansionIsExact(int n)
        {
            double[] x0 = Point(n);
            DDScalar f = Quadratic(DDScalar.Variables(x0, order: 2));

            // Exactness does not depend on the offset being small.
            foreach (double scale in new[] { 0.0, 1e-3, 0.5, 3.0, -7.5 })
            {
                double[] d = Offset(n, scale);

                double[] shifted = new double[n];
                for (int i = 0; i < n; i++) shifted[i] = x0[i] + d[i];

                double expected = Quadratic(shifted);
                double actual = f.Evaluate(d);

                Assert.True(Math.Abs(expected - actual) <= 1e-10 * (1.0 + Math.Abs(expected)),
                    $"n={n} scale={scale}: expected {expected:R}, got {actual:R}");
            }
        }

        [Theory]
        [MemberData(nameof(Sizes))]
        public void AZeroOffset_ReproducesTheValue(int n)
        {
            DDScalar f = Quadratic(DDScalar.Variables(Point(n), order: 2));

            Assert.Equal(f.Value, f.Evaluate(new double[n]), precision: 12);
        }

        /// <summary>
        /// Away from a quadratic the expansion is only an approximation, and the residual must fall
        /// like the cube of the offset. Halving the offset has to shrink the error by roughly eight.
        /// </summary>
        [Fact]
        public void OnANonQuadratic_TheErrorIsCubicInTheOffset()
        {
            var (x, y) = DDScalar.Variables(new[] { 0.6, 1.1 }, order: 2) is var v ? (v[0], v[1]) : default;
            DDScalar f = Sin(x) * Exp(y) + Log(x + y);

            static double Reference(double a, double b) => Math.Sin(a) * Math.Exp(b) + Math.Log(a + b);

            double ErrorAt(double h)
            {
                double[] d = { 0.7 * h, -0.4 * h };
                return Math.Abs(f.Evaluate(d) - Reference(0.6 + d[0], 1.1 + d[1]));
            }

            double coarse = ErrorAt(0.1);
            double fine = ErrorAt(0.05);

            Assert.True(coarse > 0.0, "the expansion should not be exact for this function");
            double ratio = coarse / fine;
            Assert.True(ratio > 6.0 && ratio < 10.0, $"expected a ratio near 8, got {ratio:R}");
        }

        [Fact]
        public void FirstOrderScalars_EvaluateTheLinearModel()
        {
            double[] x0 = Point(3);
            DDScalar f = Quadratic(DDScalar.Variables(x0, order: 1));

            double[] d = Offset(3, 0.25);

            double expected = f.Value;
            for (int i = 0; i < 3; i++) expected += f.G(i) * d[i];

            Assert.Equal(expected, f.Evaluate(d), precision: 12);
        }

        [Fact]
        public void GeneratedStructs_AgreeWithTheDynamicModel()
        {
            var (sx, sy, sz) = DD3.Variables(0.4, 0.27, 0.18);
            DD3 statik = DD3.Sin(sx) * DD3.Exp(sy) + DD3.Sqrt(sz + 1.0);

            var dyn = DDScalar.Variables(new[] { 0.4, 0.27, 0.18 }, order: 2);
            DDScalar dynamik = Sin(dyn[0]) * Exp(dyn[1]) + Sqrt(dyn[2] + 1.0);

            double[] d = { 0.11, -0.07, 0.05 };

            Assert.Equal(dynamik.Evaluate(d), statik.Evaluate(d), precision: 12);

            // The fixed-arity overload must agree with the span overload.
            Assert.Equal(statik.Evaluate(d), statik.Evaluate(0.11, -0.07, 0.05), precision: 12);
        }

        [Fact]
        public void DDScalarSpan_EvaluatesTheSameModel()
        {
            const int size = 3, order = 2;

            var dyn = DDScalar.Variables(new[] { 0.4, 0.27, 0.18 }, order);
            DDScalar dynamik = Sin(dyn[0]) * Exp(dyn[1]) + Sqrt(dyn[2] + 1.0);

            var span = new DDScalarSpan(dynamik.AsSpan(), size, order);

            double[] d = { 0.11, -0.07, 0.05 };

            Assert.Equal(dynamik.Evaluate(d), span.Evaluate(d), precision: 12);
        }

        [Fact]
        public void TheOffsetCountMustMatchTheVariableCount()
        {
            DDScalar dynamik = DDScalar.Variable(0, 1.0, size: 3);
            Assert.Throws<ArgumentException>(() => dynamik.Evaluate(new double[2]));
            Assert.Throws<ArgumentException>(() => dynamik.Evaluate(new double[4]));

            DD2 statik = DD2.Variable(0, 1.0);
            Assert.Throws<ArgumentException>(() => statik.Evaluate(new double[1]));
            Assert.Throws<ArgumentException>(() => statik.Evaluate(new double[3]));
        }

        [Fact]
        public void UninitializedDynamicScalar_Throws()
        {
            DDScalar uninitialized = default;
            Assert.Throws<InvalidOperationException>(() => uninitialized.Evaluate(new double[0]));
        }

        [Fact]
        public void Evaluate_DoesNotAllocate()
        {
            var (x, y) = DD2.Variables(3.0, 6.0);
            DD2 f = (x * y) / (x - y);

            long[] measured = new long[10];

            for (int attempt = 0; attempt < measured.Length; attempt++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();

                // Accumulate into a static so the calls cannot be optimised away. Not GC.KeepAlive:
                // that takes object and would box the double, which is itself an allocation.
                for (int i = 0; i < 1000; i++) _sink += f.Evaluate(0.01, -0.02);

                measured[attempt] = GC.GetAllocatedBytesForCurrentThread() - before;
                if (measured[attempt] == 0) return;
            }

            Assert.Fail($"Evaluate kept allocating: measured {string.Join(", ", measured)} bytes.");
        }

        private static double _sink;
    }
}
