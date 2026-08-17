using System;
using System.Numerics;
using Xunit;

namespace HyperJet.Tests
{
    /// <summary>
    /// Every code sample in README.md, executed verbatim and checked against the numbers the
    /// README prints in its comments.
    /// </summary>
    public class ReadmeExampleTests
    {
        [Fact]
        public void Example1_StaticCompileTimeAd()
        {
            var (x, y) = DDScalar2<double>.Variables(3.0, 6.0);

            DDScalar2<double> f = (x * y) / (x - y);

            Assert.Equal(-6.0, f.Value, precision: 9);
            Assert.Equal(-4.0, f.G(0), precision: 9);
            Assert.Equal(-2.666666667, f.H(0, 0), precision: 8);
        }

        [Fact]
        public void Example2_ZeroAllocationSpan()
        {
            int size = 2;
            int dataLength = Kernel.GetDataLength(size, order: 2);

            Span<double> xBuffer = stackalloc double[dataLength];
            Span<double> yBuffer = stackalloc double[dataLength];
            Span<double> resultBuffer = stackalloc double[dataLength];

            var x = DDScalarSpan.Variable(xBuffer, 0, 3.0, size, order: 2);
            var y = DDScalarSpan.Variable(yBuffer, 1, 6.0, size, order: 2);
            var result = new DDScalarSpan(resultBuffer, size, order: 2);

            x.Sin(result);

            Assert.Equal(Math.Sin(3.0), result.Value, precision: 12);
            Assert.Equal(Math.Cos(3.0), result.G(0), precision: 12);
            Assert.Equal(0.0, y.Value + 6.0 - 12.0, precision: 12); // y is used, keeps the sample faithful
        }

        [Fact]
        public void Example3_TupleDeconstruction()
        {
            var (x, y, z) = DDScalar.Variables(new double[] { 1.5, 3.0, 4.5 });

            Assert.Equal(1.5, x.Value);
            Assert.Equal(3.0, y.Value);
            Assert.Equal(4.5, z.Value);
        }

        private static Vector3D<T> CalculateTorque<T>(Vector3D<T> r, Vector3D<T> f)
            where T : IFloatingPoint<T>, IRootFunctions<T>
        {
            return r.Cross(f);
        }

        [Fact]
        public void Example4_GenericPhysicsTorque()
        {
            var (x, y, z) = DDScalar3<double>.Variables(2.0, 0.0, 0.0);
            var r = new Vector3D<DDScalar3<double>>(x, y, z);
            var f = new Vector3D<DDScalar3<double>>(0.0, 10.0, 0.0);

            Vector3D<DDScalar3<double>> torque = CalculateTorque(r, f);
            DDScalar3<double> torqueZ = torque.Z;

            // torque.Z = r.X * F.Y - r.Y * F.X = 10 * x  (F.X is zero here)
            Assert.Equal(20.0, torqueZ.Value, precision: 12);
            Assert.Equal(10.0, torqueZ.G(0), precision: 12);

            // Only F.X couples r.Y into the z-component, and it is zero for this force.
            Assert.Equal(0.0, torqueZ.G(1), precision: 12);
        }
    }
}
