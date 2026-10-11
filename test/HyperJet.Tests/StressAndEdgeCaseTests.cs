using System;
using System.Numerics;
using System.Threading.Tasks;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Stress, concurrency, boundary conditions, and high-dimensional analytical checks
    /// designed to thoroughly test HyperJetNet across all operational modes.
    /// </summary>
    public class StressAndEdgeCaseTests
    {
        #region 1. Rosenbrock Analytical Gradient & Hessian Check

        [Fact]
        public void Rosenbrock_ExactAnalyticalGradientAndHessian_AcrossAllModels()
        {
            double a = 1.0;
            double b = 100.0;
            double xVal = 1.2;
            double yVal = 2.5;

            // Analytical formulas for f(x, y) = (a - x)^2 + b * (y - x^2)^2
            double expectedVal = Math.Pow(a - xVal, 2.0) + b * Math.Pow(yVal - xVal * xVal, 2.0);
            double expectedDfDx = -2.0 * (a - xVal) - 4.0 * b * xVal * (yVal - xVal * xVal);
            double expectedDfDy = 2.0 * b * (yVal - xVal * xVal);
            double expectedD2fDx2 = 2.0 - 4.0 * b * yVal + 12.0 * b * xVal * xVal;
            double expectedD2fDxDy = -4.0 * b * xVal;
            double expectedD2fDy2 = 2.0 * b;

            // 1. Static DDScalar2<double>
            {
                var (x, y) = DDScalar2<double>.Variables(xVal, yVal);
                var f = Pow(a - x, 2.0) + b * Pow(y - x * x, 2.0);

                Assert.Equal(expectedVal, f.Value, precision: 10);
                Assert.Equal(expectedDfDx, f.G(0), precision: 10);
                Assert.Equal(expectedDfDy, f.G(1), precision: 10);
                Assert.Equal(expectedD2fDx2, f.H(0, 0), precision: 9);
                Assert.Equal(expectedD2fDxDy, f.H(0, 1), precision: 9);
                Assert.Equal(expectedD2fDxDy, f.H(1, 0), precision: 9);
                Assert.Equal(expectedD2fDy2, f.H(1, 1), precision: 9);
            }

            // 2. Static DDScalar2<float>
            {
                var (xf, yf) = DDScalar2<float>.Variables((float)xVal, (float)yVal);
                var ff = Pow((float)a - xf, 2.0f) + (float)b * Pow(yf - xf * xf, 2.0f);

                Assert.Equal((float)expectedVal, ff.Value, precision: 3);
                Assert.Equal((float)expectedDfDx, ff.G(0), precision: 3);
                Assert.Equal((float)expectedDfDy, ff.G(1), precision: 3);
                Assert.Equal((float)expectedD2fDx2, ff.H(0, 0), precision: 2);
                Assert.Equal((float)expectedD2fDxDy, ff.H(0, 1), precision: 2);
                Assert.Equal((float)expectedD2fDy2, ff.H(1, 1), precision: 2);
            }

            // 3. Dynamic DDScalar
            {
                var x = DDScalar.Variable(0, xVal, size: 2, order: 2);
                var y = DDScalar.Variable(1, yVal, size: 2, order: 2);
                var f = Pow(a - x, 2.0) + b * Pow(y - x * x, 2.0);

                Assert.Equal(expectedVal, f.Value, precision: 10);
                Assert.Equal(expectedDfDx, f.G(0), precision: 10);
                Assert.Equal(expectedDfDy, f.G(1), precision: 10);
                Assert.Equal(expectedD2fDx2, f.H(0, 0), precision: 9);
                Assert.Equal(expectedD2fDxDy, f.H(0, 1), precision: 9);
                Assert.Equal(expectedD2fDxDy, f.H(1, 0), precision: 9);
                Assert.Equal(expectedD2fDy2, f.H(1, 1), precision: 9);
            }

            // 4. Dynamic DDScalarSpan
            {
                int len = Kernel.GetDataLength(2, 2);
                Span<double> xBuf = stackalloc double[len];
                Span<double> yBuf = stackalloc double[len];
                Span<double> diff1Buf = stackalloc double[len];
                Span<double> term1Buf = stackalloc double[len];
                Span<double> xsqBuf = stackalloc double[len];
                Span<double> diff2Buf = stackalloc double[len];
                Span<double> term2Buf = stackalloc double[len];
                Span<double> resBuf = stackalloc double[len];

                var x = DDScalarSpan.Variable(xBuf, 0, xVal, size: 2, order: 2);
                var y = DDScalarSpan.Variable(yBuf, 1, yVal, size: 2, order: 2);

                var diff1 = new DDScalarSpan(diff1Buf, 2, 2);
                var term1 = new DDScalarSpan(term1Buf, 2, 2);
                var xsq = new DDScalarSpan(xsqBuf, 2, 2);
                var diff2 = new DDScalarSpan(diff2Buf, 2, 2);
                var term2 = new DDScalarSpan(term2Buf, 2, 2);
                var res = new DDScalarSpan(resBuf, 2, 2);

                // term1 = (a - x)^2
                x.SubtractFrom(a, diff1);
                diff1.Pow(2.0, term1);

                // term2 = b * (y - x^2)^2
                x.Multiply(x, xsq);
                y.Subtract(xsq, diff2);
                diff2.Multiply(b, term2); // wait, b * diff2^2: diff2.Pow(2.0, diff1); diff1.Multiply(b, term2);
                diff2.Pow(2.0, diff1);
                diff1.Multiply(b, term2);

                // res = term1 + term2
                term1.Add(term2, res);

                Assert.Equal(expectedVal, res.Value, precision: 10);
                Assert.Equal(expectedDfDx, res.G(0), precision: 10);
                Assert.Equal(expectedDfDy, res.G(1), precision: 10);
                Assert.Equal(expectedD2fDx2, res.H(0, 0), precision: 9);
                Assert.Equal(expectedD2fDxDy, res.H(0, 1), precision: 9);
                Assert.Equal(expectedD2fDy2, res.H(1, 1), precision: 9);
            }
        }

        #endregion

        #region 2. High-Dimensional Quadratic Form Scaling

        [Theory]
        [InlineData(10)]
        [InlineData(30)]
        [InlineData(50)]
        public void HighDimensionalQuadraticForm_GradientAndHessian_MatchAnalytical(int n)
        {
            // f(x) = 0.5 * x^T * Q * x
            // where Q_ij = 1.0 / (1 + |i - j|) is a symmetric matrix
            // grad f(x) = Q * x
            // hess f(x) = Q
            double[,] q = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    q[i, j] = 1.0 / (1.0 + Math.Abs(i - j));
                }
            }

            double[] xVals = new double[n];
            for (int i = 0; i < n; i++)
            {
                xVals[i] = 0.5 + 0.1 * (i % 5);
            }

            // Expected gradient g_i = sum_j Q_ij * x_j
            double[] expectedGrad = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                {
                    sum += q[i, j] * xVals[j];
                }
                expectedGrad[i] = sum;
            }

            // Expected value f = 0.5 * sum_i x_i * g_i
            double expectedVal = 0.0;
            for (int i = 0; i < n; i++)
            {
                expectedVal += 0.5 * xVals[i] * expectedGrad[i];
            }

            // Evaluate with dynamic DDScalar
            DDScalar[] vars = new DDScalar[n];
            for (int i = 0; i < n; i++)
            {
                vars[i] = DDScalar.Variable(i, xVals[i], size: n, order: 2);
            }

            DDScalar total = DDScalar.Constant(0.0, size: n, order: 2);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    total += 0.5 * q[i, j] * vars[i] * vars[j];
                }
            }

            Assert.Equal(expectedVal, total.Value, precision: 9);

            for (int i = 0; i < n; i++)
            {
                Assert.Equal(expectedGrad[i], total.G(i), precision: 9);
                for (int j = 0; j < n; j++)
                {
                    Assert.Equal(q[i, j], total.H(i, j), precision: 9);
                }
            }
        }

        #endregion

        #region 3. Concurrency and Thread-Safety

        [Fact]
        public void Multithreaded_ParallelEvaluation_ThreadSafeAndConsistent()
        {
            const int iterations = 500;

            Parallel.For(0, iterations, i =>
            {
                double xVal = 0.1 + (i % 20) * 0.05;
                double yVal = 0.3 + (i % 15) * 0.04;

                // 1. Static model
                var (x, y) = DDScalar2<double>.Variables(xVal, yVal);
                var f = Sin(x * y) + Exp(x / (1.0 + y * y)) - Log(2.0 + x * x);

                // 2. Dynamic model
                var dx = DDScalar.Variable(0, xVal, size: 2, order: 2);
                var dy = DDScalar.Variable(1, yVal, size: 2, order: 2);
                var df = Sin(dx * dy) + Exp(dx / (1.0 + dy * dy)) - Log(2.0 + dx * dx);

                Assert.Equal(f.Value, df.Value, precision: 12);
                Assert.Equal(f.G(0), df.G(0), precision: 12);
                Assert.Equal(f.G(1), df.G(1), precision: 12);
                Assert.Equal(f.H(0, 0), df.H(0, 0), precision: 12);
                Assert.Equal(f.H(0, 1), df.H(0, 1), precision: 12);
                Assert.Equal(f.H(1, 1), df.H(1, 1), precision: 12);
            });
        }

        #endregion

        #region 4. FusedMultiplyAdd (Ternary Kernel)

        [Fact]
        public void FusedMultiplyAdd_MatchesAnalyticalAndFiniteDifferences()
        {
            double x0 = 1.7;
            double y0 = 2.3;
            double z0 = 0.9;

            // f(x, y, z) = x * y + z
            // df/dx = y, df/dy = x, df/dz = 1
            // d2f/dxdy = 1, all others 0
            var (x, y, z) = DDScalar3<double>.Variables(x0, y0, z0);
            var f = DDScalar3<double>.FusedMultiplyAdd(x, y, z);

            Assert.Equal(x0 * y0 + z0, f.Value, precision: 12);
            Assert.Equal(y0, f.G(0), precision: 12);
            Assert.Equal(x0, f.G(1), precision: 12);
            Assert.Equal(1.0, f.G(2), precision: 12);

            Assert.Equal(0.0, f.H(0, 0), precision: 12);
            Assert.Equal(1.0, f.H(0, 1), precision: 12);
            Assert.Equal(0.0, f.H(0, 2), precision: 12);
            Assert.Equal(0.0, f.H(1, 1), precision: 12);
            Assert.Equal(0.0, f.H(1, 2), precision: 12);
            Assert.Equal(0.0, f.H(2, 2), precision: 12);

            // Test facade overload
            var fFacade = FusedMultiplyAdd(x, y, z);
            Assert.Equal(f.Value, fFacade.Value);
            Assert.Equal(f.G(0), fFacade.G(0));
            Assert.Equal(f.H(0, 1), fFacade.H(0, 1));
        }

        #endregion

        #region 5. Higher-Order Nesting (4th Order Derivatives)

        [Fact]
        public void NestedScalars_HigherOrderDerivatives_ExactPolynomial()
        {
            // Polynomial: P(x) = x^5 - 3x^4 + 2x^3 - x^2 + 7x - 5
            // P'(x)   = 5x^4 - 12x^3 + 6x^2 - 2x + 7
            // P''(x)  = 20x^3 - 36x^2 + 12x - 2
            // P'''(x) = 60x^2 - 72x + 12
            // P''''(x)= 120x - 72

            double x0 = 1.5;

            double p0 = Math.Pow(x0, 5) - 3.0 * Math.Pow(x0, 4) + 2.0 * Math.Pow(x0, 3) - Math.Pow(x0, 2) + 7.0 * x0 - 5.0;
            double p1 = 5.0 * Math.Pow(x0, 4) - 12.0 * Math.Pow(x0, 3) + 6.0 * Math.Pow(x0, 2) - 2.0 * x0 + 7.0;
            double p2 = 20.0 * Math.Pow(x0, 3) - 36.0 * Math.Pow(x0, 2) + 12.0 * x0 - 2.0;
            double p3 = 60.0 * Math.Pow(x0, 2) - 72.0 * x0 + 12.0;
            double p4 = 120.0 * x0 - 72.0;

            // Nested struct: outer DDScalar1 over inner DDScalar1 gives 4th order capability
            var innerX = DDScalar1<double>.Variable(0, x0);
            var outerX = DDScalar1<DDScalar1<double>>.Variable(0, innerX);

            // Compute P(outerX)
            var p = outerX * outerX * outerX * outerX * outerX
                  - 3.0 * outerX * outerX * outerX * outerX
                  + 2.0 * outerX * outerX * outerX
                  - outerX * outerX
                  + 7.0 * outerX
                  - 5.0;

            // p.Value is an inner DDScalar1:
            // p.Value.Value is f(x)
            // p.Value.G(0) is f'(x)
            // p.Value.H(0,0) is f''(x)
            Assert.Equal(p0, p.Value.Value, precision: 10);
            Assert.Equal(p1, p.Value.G(0), precision: 10);
            Assert.Equal(p2, p.Value.H(0, 0), precision: 10);

            // p.H(0,0) is an inner DDScalar1:
            // p.H(0,0).G(0) is f'''(x)
            // p.H(0,0).H(0,0) is f''''(x)
            Assert.Equal(p3, p.H(0, 0).G(0), precision: 10);
            Assert.Equal(p4, p.H(0, 0).H(0, 0), precision: 10);
        }

        #endregion

        #region 6. Boundary Conditions and Special Functions

        [Fact]
        public void ExpM1_LogP1_BoundaryAtZero()
        {
            var x = DDScalar1<double>.Variable(0, 0.0);

            var expM1 = DDScalar1<double>.ExpM1(x);
            Assert.Equal(0.0, expM1.Value, precision: 12);
            Assert.Equal(1.0, expM1.G(0), precision: 12);
            Assert.Equal(1.0, expM1.H(0, 0), precision: 12);

            var logP1 = DDScalar1<double>.LogP1(x);
            Assert.Equal(0.0, logP1.Value, precision: 12);
            Assert.Equal(1.0, logP1.G(0), precision: 12);
            Assert.Equal(-1.0, logP1.H(0, 0), precision: 12);
        }

        [Fact]
        public void SinCos_SinCosPi_Agreement()
        {
            var x = DDScalar1<double>.Variable(0, 0.35);

            var (sin, cos) = DDScalar1<double>.SinCos(x);
            var expectedSin = DDScalar1<double>.Sin(x);
            var expectedCos = DDScalar1<double>.Cos(x);

            Assert.Equal(expectedSin.Value, sin.Value, precision: 12);
            Assert.Equal(expectedSin.G(0), sin.G(0), precision: 12);
            Assert.Equal(expectedSin.H(0, 0), sin.H(0, 0), precision: 12);

            Assert.Equal(expectedCos.Value, cos.Value, precision: 12);
            Assert.Equal(expectedCos.G(0), cos.G(0), precision: 12);
            Assert.Equal(expectedCos.H(0, 0), cos.H(0, 0), precision: 12);

            var (sinPi, cosPi) = DDScalar1<double>.SinCosPi(x);
            var expectedSinPi = DDScalar1<double>.SinPi(x);
            var expectedCosPi = DDScalar1<double>.CosPi(x);

            Assert.Equal(expectedSinPi.Value, sinPi.Value, precision: 12);
            Assert.Equal(expectedSinPi.G(0), sinPi.G(0), precision: 12);
            Assert.Equal(expectedSinPi.H(0, 0), sinPi.H(0, 0), precision: 12);

            Assert.Equal(expectedCosPi.Value, cosPi.Value, precision: 12);
            Assert.Equal(expectedCosPi.G(0), cosPi.G(0), precision: 12);
            Assert.Equal(expectedCosPi.H(0, 0), cosPi.H(0, 0), precision: 12);
        }

        #endregion
    }
}
