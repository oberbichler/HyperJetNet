global using DDScalar2 = HyperJet.DDScalar2<double>;

using Xunit;
using System.Numerics;
using HyperJet;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    public class DDScalarStaticTests
    {
        private const double Tolerance = 1e-9;

        [Fact]
        public void DDScalar2_QuickstartExample_ShouldMatchAnalyticalDerivatives()
        {
            // Arrange
            var (x, y) = DDScalar2.Variables(3.0, 6.0);

            // Act
            DDScalar2 f = (x * y) / (x - y);

            // Assert
            Assert.Equal(-6.0, f.Value, precision: 9);

            // First-order derivatives (gradient)
            Assert.Equal(-4.0, f.G(0), precision: 9); // df/dx
            Assert.Equal(1.0, f.G(1), precision: 9);  // df/dy

            // Second-order derivatives (Hessian)
            Assert.Equal(-8.0 / 3.0, f.H(0, 0), precision: 9);  // d²f/dx²
            Assert.Equal(4.0 / 3.0, f.H(0, 1), precision: 9);   // d²f/dxdy
            Assert.Equal(4.0 / 3.0, f.H(1, 0), precision: 9);   // Symmetric d²f/dydx
            Assert.Equal(-2.0 / 3.0, f.H(1, 1), precision: 9);  // d²f/dy²
        }

        [Fact]
        public void HyperJetMath_SinCos_ShouldMatchAnalytical()
        {
            // f(x, y) = sin(x) * cos(y)
            var (x, y) = DDScalar2.Variables(1.0, 2.0);
            DDScalar2 f = Sin(x) * Cos(y);

            double expectedVal = Math.Sin(1.0) * Math.Cos(2.0);
            double expectedDfDx = Math.Cos(1.0) * Math.Cos(2.0);
            double expectedDfDy = -Math.Sin(1.0) * Math.Sin(2.0);
            double expectedD2fDx2 = -Math.Sin(1.0) * Math.Cos(2.0);
            double expectedD2fDxDy = -Math.Cos(1.0) * Math.Sin(2.0);
            double expectedD2fDy2 = -Math.Sin(1.0) * Math.Cos(2.0);

            Assert.Equal(expectedVal, f.Value, precision: 9);
            Assert.Equal(expectedDfDx, f.G(0), precision: 9);
            Assert.Equal(expectedDfDy, f.G(1), precision: 9);
            Assert.Equal(expectedD2fDx2, f.H(0, 0), precision: 9);
            Assert.Equal(expectedD2fDxDy, f.H(0, 1), precision: 9);
            Assert.Equal(expectedD2fDy2, f.H(1, 1), precision: 9);
        }

        [Fact]
        public void HyperJetMath_Atan2_ShouldMatchAnalytical()
        {
            // f(y, x) = atan2(y, x)
            var (y, x) = DDScalar2.Variables(2.0, 3.0);
            DDScalar2 f = Atan2(y, x);

            double tmp = 2.0 * 2.0 + 3.0 * 3.0; // y^2 + x^2 = 13
            double expectedVal = Math.Atan2(2.0, 3.0);
            double expectedDfDy = 3.0 / tmp;  // x / tmp = 3 / 13
            double expectedDfDx = -2.0 / tmp; // -y / tmp = -2 / 13

            // Second derivatives
            double expectedD2fDy2 = (-2.0 / tmp) * (3.0 / tmp) * 2.0; // db * da * 2 = (-2/13) * (3/13) * 2 = -12 / 169
            double expectedD2fDyDx = (-2.0 / tmp) * (-2.0 / tmp) - (3.0 / tmp) * (3.0 / tmp); // db^2 - da^2 = 4/169 - 9/169 = -5 / 169

            Assert.Equal(expectedVal, f.Value, precision: 9);
            Assert.Equal(expectedDfDy, f.G(0), precision: 9);
            Assert.Equal(expectedDfDx, f.G(1), precision: 9);
            Assert.Equal(expectedD2fDy2, f.H(0, 0), precision: 9);
            Assert.Equal(expectedD2fDyDx, f.H(0, 1), precision: 9);
        }

        [Fact]
        public void DDScalar2_ConstantAndVariables_ShouldEvaluateCorrectly()
        {
            var x = DDScalar2.Variable(0, 5.0);
            var y = DDScalar2.Variable(1, 10.0);

            Assert.Equal(5.0, x.Value);
            Assert.Equal(1.0, x.G(0));
            Assert.Equal(0.0, x.G(1));

            Assert.Equal(10.0, y.Value);
            Assert.Equal(0.0, y.G(0));
            Assert.Equal(1.0, y.G(1));

            var c = DDScalar2.Constant(42.0);
            Assert.Equal(42.0, c.Value);
            Assert.Equal(0.0, c.G(0));
            Assert.Equal(0.0, c.G(1));
        }

        [Fact]
        public void HyperJetMath_PowerAndSqrt_ShouldMatchAnalytical()
        {
            var (x, y) = DDScalar2.Variables(4.0, 9.0);

            DDScalar2 f1 = Sqrt(x);
            Assert.Equal(2.0, f1.Value, precision: 9);
            Assert.Equal(0.25, f1.G(0), precision: 9);
            Assert.Equal(-1.0 / 32.0, f1.H(0, 0), precision: 9);

            DDScalar2 f2 = Pow(y, 1.5);
            Assert.Equal(27.0, f2.Value, precision: 9);
            Assert.Equal(1.5 * 3.0, f2.G(1), precision: 9);
            Assert.Equal(0.75 / 3.0, f2.H(1, 1), precision: 9);
        }

        [Fact]
        public void GenericMath_IFloatingPoint_ShouldEvaluateCorrectly()
        {
            var (x, y) = DDScalar2.Variables(3.0, 6.0);

            DDScalar2 result = CalculateGeneric(x, y);

            Assert.Equal(-6.0, result.Value, precision: 9);
            Assert.Equal(-4.0, result.G(0), precision: 9); // df/dx
            Assert.Equal(1.0, result.G(1), precision: 9);  // df/dy
        }

        private static T CalculateGeneric<T>(T x, T y) where T : IFloatingPoint<T>
        {
            return (x * y) / (x - y);
        }

        [Fact]
        public void GenericMath_Trigonometric_ShouldEvaluateCorrectly()
        {
            var x = DDScalar2.Variable(0, 1.0);

            DDScalar2 result = CalculateTrig(x);

            DDScalar2 expected = Sin(x) * Cos(x);

            Assert.Equal(expected.Value, result.Value, precision: 9);
            Assert.Equal(expected.G(0), result.G(0), precision: 9);
            Assert.Equal(expected.H(0, 0), result.H(0, 0), precision: 9);
        }

        private static T CalculateTrig<T>(T x) where T : IFloatingPoint<T>, ITrigonometricFunctions<T>
        {
            return T.Sin(x) * T.Cos(x);
        }

        [Fact]
        public void GenericMath_Constants_ShouldBeCorrect()
        {
            Assert.Equal(Math.PI, DDScalar2.Pi.Value);
            Assert.Equal(Math.E, DDScalar2.E.Value);
            Assert.Equal(Math.Tau, DDScalar2.Tau.Value);
        }

        [Fact]
        public void GenericMath_Modulo_ShouldEvaluateCorrectly()
        {
            var (x, y) = DDScalar2.Variables(5.0, 3.0);
            DDScalar2 result = x % y;

            Assert.Equal(2.0, result.Value);
            // x % y = x - y * floor(x/y) = x - y * 1 = x - y
            // dx/dx = 1, dx/dy = -1
            Assert.Equal(1.0, result.G(0));
            Assert.Equal(-1.0, result.G(1));
        }

        [Fact]
        public void DDScalar2_FloatSupport_ShouldEvaluateCorrectly()
        {
            var (x, y) = DDScalar2<float>.Variables(3.0f, 6.0f);
            DDScalar2<float> f = (x * y) / (x - y);

            Assert.Equal(-6.0f, f.Value);
            Assert.Equal(-4.0f, f.G(0));
            Assert.Equal(1.0f, f.G(1));
            Assert.Equal(-8.0f / 3.0f, f.H(0, 0), precision: 5);
        }

        [Fact]
        public void DDScalarN_GeneratedDimensions_ShouldWorkCorrectly()
        {
            // Test DDScalar1
            var x1 = DDScalar1<double>.Variables(3.0);
            DDScalar1<double> result1 = x1 * x1;
            Assert.Equal(9.0, result1.Value);
            Assert.Equal(6.0, result1.G(0)); // d(x^2)/dx = 2x = 6

            // Test DDScalar3
            var (x3, y3, z3) = DDScalar3<double>.Variables(1.0, 2.0, 3.0);
            DDScalar3<double> result3 = x3 * y3 * z3;
            Assert.Equal(6.0, result3.Value);
            Assert.Equal(6.0, result3.G(0)); // dyz = 6
            Assert.Equal(3.0, result3.G(1)); // dxz = 3
            Assert.Equal(2.0, result3.G(2)); // dxy = 2

            // Test DDScalar15
            var vars15 = DDScalar15<double>.Variables(
                1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0, 15.0);
            DDScalar15<double> sum15 = vars15.v1 + vars15.v15;
            Assert.Equal(16.0, sum15.Value);
            Assert.Equal(1.0, sum15.G(0));
            Assert.Equal(1.0, sum15.G(14));
        }

        [Fact]
        public void Vector3D_GenericOperations_ShouldEvaluateCorrectly()
        {
            // 1. Test with double
            var v1 = new Vector3D<double>(1.0, 2.0, 3.0);
            var v2 = new Vector3D<double>(4.0, 5.0, 6.0);

            var vAdd = v1 + v2;
            Assert.Equal(5.0, vAdd.X);
            Assert.Equal(7.0, vAdd.Y);
            Assert.Equal(9.0, vAdd.Z);

            double dot = Vector3D<double>.Dot(v1, v2);
            Assert.Equal(1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0, dot); // 4 + 10 + 18 = 32

            var cross = Vector3D<double>.Cross(v1, v2);
            Assert.Equal(2.0 * 6.0 - 3.0 * 5.0, cross.X); // -3
            Assert.Equal(3.0 * 4.0 - 1.0 * 6.0, cross.Y); // 6
            Assert.Equal(1.0 * 5.0 - 2.0 * 4.0, cross.Z); // -3

            // 2. Test with DDScalar3<double> to verify automatic differentiation on vectors
            var (x, y, z) = DDScalar3<double>.Variables(1.0, 2.0, 3.0);
            var u = new Vector3D<DDScalar3<double>>(x, y, z);
            var v = new Vector3D<DDScalar3<double>>(
                DDScalar3<double>.Constant(4.0),
                DDScalar3<double>.Constant(5.0),
                DDScalar3<double>.Constant(6.0));

            // Dot product = x*4 + y*5 + z*6
            DDScalar3<double> uDotV = u.Dot(v);
            Assert.Equal(32.0, uDotV.Value);
            Assert.Equal(4.0, uDotV.G(0)); // d(Dot)/dx = 4
            Assert.Equal(5.0, uDotV.G(1)); // d(Dot)/dy = 5
            Assert.Equal(6.0, uDotV.G(2)); // d(Dot)/dz = 6

            // Length
            DDScalar3<double> len = u.Length();
            Assert.Equal(Math.Sqrt(1.0 + 4.0 + 9.0), len.Value);

            // 3. Test mixing components using implicit conversion from double to DDScalar3<double>
            var w = new Vector3D<DDScalar3<double>>();
            w.X = x;     // DDScalar3<double> variable
            w.Y = 10.0;  // Implicitly converted to DDScalar3<double> Constant(10.0)!
            w.Z = z;     // DDScalar3<double> variable

            Assert.Equal(1.0, w.X.Value);
            Assert.Equal(10.0, w.Y.Value);
            Assert.Equal(0.0, w.Y.G(0)); // Constant has 0 derivatives
            Assert.Equal(3.0, w.Z.Value);
        }
    }
}
