using Xunit;
using HyperJet;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    public class DDScalarTests
    {
        [Fact]
        public void Variables_FirstOrder_ShouldComputeCorrectly()
        {
            // Set up a 1st order calculation with 3 variables
            double[] values = { 1.0, 2.0, 3.0 };
            DDScalar[] vars = DDScalar.Variables(values, order: 1);

            Assert.Equal(3, vars.Length);
            Assert.Equal(1, vars[0].Order);

            DDScalar x = vars[0];
            DDScalar y = vars[1];
            DDScalar z = vars[2];

            // f(x, y, z) = x * y + z
            DDScalar f = x * y + z;

            Assert.Equal(5.0, f.Value);
            Assert.Equal(2.0, f.G(0)); // df/dx = y = 2
            Assert.Equal(1.0, f.G(1)); // df/dy = x = 1
            Assert.Equal(1.0, f.G(2)); // df/dz = 1

            // Hessian should throw for 1st order
            Assert.Throws<InvalidOperationException>(() => f.H(0, 0));
        }

        [Fact]
        public void Variables_SecondOrder_ShouldComputeCorrectly()
        {
            // Set up a 2nd order calculation with 3 variables
            double[] values = { 3.0, 6.0 };
            DDScalar[] vars = DDScalar.Variables(values, order: 2);

            DDScalar x = vars[0];
            DDScalar y = vars[1];

            // f(x, y) = (x * y) / (x - y)
            DDScalar f = (x * y) / (x - y);

            Assert.Equal(-6.0, f.Value, precision: 9);
            Assert.Equal(-4.0, f.G(0), precision: 9);
            Assert.Equal(1.0, f.G(1), precision: 9);

            Assert.Equal(-8.0 / 3.0, f.H(0, 0), precision: 9);
            Assert.Equal(4.0 / 3.0, f.H(0, 1), precision: 9);
            Assert.Equal(-2.0 / 3.0, f.H(1, 1), precision: 9);
        }

        [Fact]
        public void Variables_IncompatibleSettings_ShouldThrowInvalidOperationException()
        {
            DDScalar a = DDScalar.Variable(0, 2.0, size: 2, order: 2);
            DDScalar b = DDScalar.Variable(1, 3.0, size: 3, order: 2); // different size

            Assert.Throws<InvalidOperationException>(() => a + b);

            DDScalar c = DDScalar.Variable(0, 2.0, size: 2, order: 1); // different order
            Assert.Throws<InvalidOperationException>(() => a + c);
        }

        [Fact]
        public void DDScalarSpan_ZeroAllocation_ShouldComputeCorrectly()
        {
            int size = 2;
            int order = 2;
            int dataLength = Kernel.GetDataLength(size, order); // 6

            // Allocate buffers on stack (or array for test context)
            Span<double> xBuffer = stackalloc double[dataLength];
            Span<double> yBuffer = stackalloc double[dataLength];
            Span<double> mulBuffer = stackalloc double[dataLength];
            Span<double> subBuffer = stackalloc double[dataLength];
            Span<double> destBuffer = stackalloc double[dataLength];

            // Initialize x and y as variables
            var x = DDScalarSpan.Variable(xBuffer, 0, 3.0, size, order);
            var y = DDScalarSpan.Variable(yBuffer, 1, 6.0, size, order);

            var mul = new DDScalarSpan(mulBuffer, size, order);
            var sub = new DDScalarSpan(subBuffer, size, order);
            var dest = new DDScalarSpan(destBuffer, size, order);

            // Compute f(x, y) = (x * y) / (x - y) step-by-step
            x.Multiply(y, mul);       // mul = x * y
            x.Subtract(y, sub);       // sub = x - y
            mul.Divide(sub, dest);    // dest = mul / sub

            // Verify values and derivatives
            Assert.Equal(-6.0, dest.Value, precision: 9);
            Assert.Equal(-4.0, dest.G(0), precision: 9);
            Assert.Equal(1.0, dest.G(1), precision: 9);

            Assert.Equal(-8.0 / 3.0, dest.H(0, 0), precision: 9);
            Assert.Equal(4.0 / 3.0, dest.H(0, 1), precision: 9);
            Assert.Equal(-2.0 / 3.0, dest.H(1, 1), precision: 9);
        }

        [Fact]
        public void DDScalarSpan_Operators_ShouldEvaluateCorrectly()
        {
            int size = 2;
            int order = 2;
            int dataLength = Kernel.GetDataLength(size, order);

            Span<double> xBuffer = stackalloc double[dataLength];
            Span<double> yBuffer = stackalloc double[dataLength];

            var x = DDScalarSpan.Variable(xBuffer, 0, 3.0, size, order);
            var y = DDScalarSpan.Variable(yBuffer, 1, 6.0, size, order);

            // Using standard operator fallback to DDScalar (heap allocation but convenient)
            DDScalar f = (x * y) / (x - y);

            Assert.Equal(-6.0, f.Value, precision: 9);
            Assert.Equal(-4.0, f.G(0), precision: 9);
            Assert.Equal(1.0, f.G(1), precision: 9);

            Assert.Equal(-8.0 / 3.0, f.H(0, 0), precision: 9);
            Assert.Equal(4.0 / 3.0, f.H(0, 1), precision: 9);
            Assert.Equal(-2.0 / 3.0, f.H(1, 1), precision: 9);
        }

        [Fact]
        public void TupleDeconstruction_ArrayAndSpans_ShouldWorkCorrectly()
        {
            // Test array deconstruction for DDScalar
            var (x, y, z) = DDScalar.Variables(new double[] { 1.0, 2.0, 3.0 });
            Assert.Equal(1.0, x.Value);
            Assert.Equal(2.0, y.Value);
            Assert.Equal(3.0, z.Value);
            Assert.Equal(1.0, x.G(0));
            Assert.Equal(1.0, y.G(1));
            Assert.Equal(1.0, z.G(2));

            // Test span deconstruction
            Span<double> span = stackalloc double[] { 10.0, 20.0, 30.0, 40.0, 50.0 };
            var (v1, v2, v3, v4, v5) = span;
            Assert.Equal(10.0, v1);
            Assert.Equal(20.0, v2);
            Assert.Equal(30.0, v3);
            Assert.Equal(40.0, v4);
            Assert.Equal(50.0, v5);

            // Test ReadOnlySpan deconstruction
            ReadOnlySpan<int> roSpan = new int[] { 100, 200 };
            var (a, b) = roSpan;
            Assert.Equal(100, a);
            Assert.Equal(200, b);
        }

        [Fact]
        public void Export_VectorAndMatrix_ShouldWorkCorrectly()
        {
            // 1. Test DDScalar
            double[] values = { 3.0, 6.0 };
            DDScalar[] vars = DDScalar.Variables(values, order: 2);
            DDScalar f = (vars[0] * vars[1]) / (vars[0] - vars[1]);

            double[] fGrad = f.GetGradient();
            Assert.Equal(2, fGrad.Length);
            Assert.Equal(-4.0, fGrad[0], precision: 9);
            Assert.Equal(1.0, fGrad[1], precision: 9);

            double[] destGrad = new double[2];
            f.GetGradient(destGrad);
            Assert.Equal(-4.0, destGrad[0], precision: 9);
            Assert.Equal(1.0, destGrad[1], precision: 9);

            double[,] fHess = f.GetHessian();
            Assert.Equal(2, fHess.GetLength(0));
            Assert.Equal(2, fHess.GetLength(1));
            Assert.Equal(-8.0 / 3.0, fHess[0, 0], precision: 9);
            Assert.Equal(4.0 / 3.0, fHess[0, 1], precision: 9);
            Assert.Equal(4.0 / 3.0, fHess[1, 0], precision: 9);
            Assert.Equal(-2.0 / 3.0, fHess[1, 1], precision: 9);

            // 2. Test DDScalarSpan
            int size = 2;
            int order = 2;
            int dataLength = Kernel.GetDataLength(size, order);
            Span<double> xBuffer = stackalloc double[dataLength];
            Span<double> yBuffer = stackalloc double[dataLength];
            var xSpan = DDScalarSpan.Variable(xBuffer, 0, 3.0, size, order);
            var ySpan = DDScalarSpan.Variable(yBuffer, 1, 6.0, size, order);
            DDScalar fSpan = (xSpan * ySpan) / (xSpan - ySpan);

            double[] fSpanGrad = fSpan.GetGradient();
            Assert.Equal(-4.0, fSpanGrad[0], precision: 9);
            Assert.Equal(1.0, fSpanGrad[1], precision: 9);

            double[,] fSpanHess = fSpan.GetHessian();
            Assert.Equal(-8.0 / 3.0, fSpanHess[0, 0], precision: 9);
            Assert.Equal(4.0 / 3.0, fSpanHess[0, 1], precision: 9);

            // 3. Test static DDScalar2
            var (xs, ys) = DDScalar2.Variables(3.0, 6.0);
            DDScalar2 fs = (xs * ys) / (xs - ys);

            double[] fsGrad = fs.GetGradient();
            Assert.Equal(-4.0, fsGrad[0], precision: 9);
            Assert.Equal(1.0, fsGrad[1], precision: 9);

            double[,] fsHess = fs.GetHessian();
            Assert.Equal(-8.0 / 3.0, fsHess[0, 0], precision: 9);
            Assert.Equal(4.0 / 3.0, fsHess[0, 1], precision: 9);
        }

        [Fact]
        public void HyperJetMath_NewMathFunctions_ShouldEvaluateCorrectly()
        {
            // 1. Test DDScalar brand new math methods (Sinh, Cosh, Tanh, Cbrt, Log2)
            double[] values = { 1.5 };
            DDScalar a = DDScalar.Variables(values, order: 2)[0];

            DDScalar sh = Sinh(a);
            Assert.Equal(Math.Sinh(1.5), sh.Value, precision: 9);
            Assert.Equal(Math.Cosh(1.5), sh.G(0), precision: 9);
            Assert.Equal(Math.Sinh(1.5), sh.H(0, 0), precision: 9);

            DDScalar ch = Cosh(a);
            Assert.Equal(Math.Cosh(1.5), ch.Value, precision: 9);
            Assert.Equal(Math.Sinh(1.5), ch.G(0), precision: 9);
            Assert.Equal(Math.Cosh(1.5), ch.H(0, 0), precision: 9);

            DDScalar th = Tanh(a);
            Assert.Equal(Math.Tanh(1.5), th.Value, precision: 9);
            double expectedTanhDeriv = 1.0 - Math.Tanh(1.5) * Math.Tanh(1.5);
            Assert.Equal(expectedTanhDeriv, th.G(0), precision: 9);

            DDScalar cb = Cbrt(a);
            Assert.Equal(Math.Cbrt(1.5), cb.Value, precision: 9);

            DDScalar lg2 = Log2(a);
            Assert.Equal(Math.Log2(1.5), lg2.Value, precision: 9);

            // 2. Test DDScalarSpan zero-allocation transcendent methods
            int size = 1;
            int order = 2;
            int dataLength = Kernel.GetDataLength(size, order);
            Span<double> bufA = stackalloc double[dataLength];
            Span<double> bufDest = stackalloc double[dataLength];

            var aSpan = DDScalarSpan.Variable(bufA, 0, 1.5, size, order);
            var destSpan = new DDScalarSpan(bufDest, size, order);

            // Test Sin
            aSpan.Sin(destSpan);
            Assert.Equal(Math.Sin(1.5), destSpan.Value, precision: 9);
            Assert.Equal(Math.Cos(1.5), destSpan.G(0), precision: 9);
            Assert.Equal(-Math.Sin(1.5), destSpan.H(0, 0), precision: 9);

            // Test Exp
            aSpan.Exp(destSpan);
            Assert.Equal(Math.Exp(1.5), destSpan.Value, precision: 9);

            // Test Sinh
            aSpan.Sinh(destSpan);
            Assert.Equal(Math.Sinh(1.5), destSpan.Value, precision: 9);
        }
    }
}
