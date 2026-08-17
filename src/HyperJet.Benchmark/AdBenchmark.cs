using BenchmarkDotNet.Attributes;
using HyperJet;
using System;

namespace HyperJet.Benchmark
{
    [MemoryDiagnoser]
    [SimpleJob(launchCount: 1, warmupCount: 2, iterationCount: 3)]
    public class AdBenchmark
    {
        private const double XVal = 3.0;
        private const double YVal = 6.0;

        [Params(2, 10)]
        public int Size { get; set; }

        [Benchmark(Baseline = true)]
        public double StaticDouble_DDScalar2()
        {
            // Note: DDScalar2 has a fixed Size=2. For Size=10, we still run it to show 2-variable baseline.
            var (x, y) = DDScalar2<double>.Variables(XVal, YVal);
            DDScalar2<double> f = (x * y) / (x - y);
            return f.Value + f.G(0) + f.H(0, 1);
        }

        [Benchmark]
        public float StaticFloat_DDScalar2()
        {
            var (x, y) = DDScalar2<float>.Variables((float)XVal, (float)YVal);
            DDScalar2<float> f = (x * y) / (x - y);
            return f.Value + f.G(0) + f.H(0, 1);
        }

        // The same arithmetic over ten compile-time variables, once with a Hessian and once without.
        // Only two of the ten take part, so the expression is identical and what differs is the
        // storage -- 66 coefficients against 11 -- and the kernel work that goes with it.

        [Benchmark]
        public double StaticSecondOrder_DDScalar10()
        {
            var x = DDScalar10<double>.Variable(0, XVal);
            var y = DDScalar10<double>.Variable(1, YVal);
            DDScalar10<double> f = (x * y) / (x - y);
            return f.Value + f.G(0);
        }

        [Benchmark]
        public double StaticFirstOrder_DScalar10()
        {
            var x = DScalar10<double>.Variable(0, XVal);
            var y = DScalar10<double>.Variable(1, YVal);
            DScalar10<double> f = (x * y) / (x - y);
            return f.Value + f.G(0);
        }

        [Benchmark]
        public double DynamicHeap_DDScalar()
        {
            DDScalar x = DDScalar.Variable(0, XVal, size: Size, order: 2);
            DDScalar y = DDScalar.Variable(1, YVal, size: Size, order: 2);
            DDScalar f = (x * y) / (x - y);
            return f.Value + f.G(0) + f.H(0, 1);
        }

        [Benchmark]
        public double DynamicStack_DDScalarSpan()
        {
            int length = Kernel.GetDataLength(Size, 2);

            Span<double> xBuffer = stackalloc double[length];
            Span<double> yBuffer = stackalloc double[length];
            Span<double> mulBuffer = stackalloc double[length];
            Span<double> subBuffer = stackalloc double[length];
            Span<double> destBuffer = stackalloc double[length];

            var x = DDScalarSpan.Variable(xBuffer, 0, XVal, Size, 2);
            var y = DDScalarSpan.Variable(yBuffer, 1, YVal, Size, 2);

            var mul = new DDScalarSpan(mulBuffer, Size, 2);
            var sub = new DDScalarSpan(subBuffer, Size, 2);
            var dest = new DDScalarSpan(destBuffer, Size, 2);

            x.Multiply(y, mul);
            x.Subtract(y, sub);
            mul.Divide(sub, dest);

            return dest.Value + dest.G(0) + dest.H(0, 1);
        }

        [Benchmark]
        public double DynamicHeap_ScalarOnly_NoSIMD()
        {
            int length = Kernel.GetDataLength(Size, 2);
            double[] x = new double[length];
            double[] y = new double[length];
            double[] mul = new double[length];
            double[] sub = new double[length];
            double[] dest = new double[length];

            x[0] = XVal;
            x[1 + 0] = 1.0;

            y[0] = YVal;
            y[1 + 1] = 1.0;

            ScalarMultiply(x, y, mul, Size, 2);
            ScalarSubtract(x, y, sub, Size, 2);
            ScalarDivide(mul, sub, dest, Size, 2);

            return dest[0] + dest[1] + dest[1 + Size];
        }

        #region Scalar Fallback Methods for SIMD Comparison

        private static void ScalarSubtract(double[] a, double[] b, double[] r, int size, int order)
        {
            r[0] = a[0] - b[0];
            int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
            for (int i = 1; i < n; i++)
            {
                r[i] = a[i] - b[i];
            }
        }

        private static void ScalarMultiply(double[] a, double[] b, double[] r, int size, int order)
        {
            r[0] = a[0] * b[0];
            int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
            for (int i = 1; i < n; i++)
            {
                r[i] = a[i] * b[0] + b[i] * a[0];
            }

            if (order >= 2)
            {
                int k = 1 + size;
                for (int i = 0; i < size; i++)
                {
                    double ai = a[1 + i];
                    double bi = b[1 + i];
                    for (int j = i; j < size; j++)
                    {
                        r[k++] += bi * a[1 + j] + ai * b[1 + j];
                    }
                }
            }
        }

        private static void ScalarDivide(double[] a, double[] b, double[] r, int size, int order)
        {
            double tmp = 1.0 / b[0];
            double f = a[0] * tmp;
            r[0] = f;

            double da = tmp;
            double db = -a[0] * tmp * tmp;

            int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
            for (int i = 1; i < n; i++)
            {
                r[i] = a[i] * da + b[i] * db;
            }

            if (order >= 2)
            {
                double dab = -tmp * tmp;
                double dbb = 2.0 * a[0] * tmp * tmp * tmp;

                int k = 1 + size;
                for (int i = 0; i < size; i++)
                {
                    double ai = a[1 + i];
                    double bi = b[1 + i];
                    double ca = bi * dab;
                    double cb = ai * dab + bi * dbb;

                    for (int j = i; j < size; j++)
                    {
                        r[k++] += ca * a[1 + j] + cb * b[1 + j];
                    }
                }
            }
        }

        #endregion
    }
}
