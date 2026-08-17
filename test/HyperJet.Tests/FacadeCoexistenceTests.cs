using Xunit;
using static System.Math;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// The facade contributes 571 generic overloads named <c>Sin</c>, <c>Log</c>, <c>Abs</c> and so on.
    /// Importing it next to <c>System.Math</c> must stay unambiguous: plain numeric arguments have to
    /// keep resolving to <c>System.Math</c>. This file compiles with both static imports, so a
    /// regression here is a build error rather than a silent behaviour change.
    /// </summary>
    public class FacadeCoexistenceTests
    {
        [Fact]
        public void PlainDoubles_StillResolveToSystemMath()
        {
            Assert.Equal(System.Math.Sin(0.7), Sin(0.7));
            Assert.Equal(System.Math.Cos(0.7), Cos(0.7));
            Assert.Equal(System.Math.Sqrt(2.0), Sqrt(2.0));
            Assert.Equal(System.Math.Log(2.0, 10.0), Log(2.0, 10.0));
            Assert.Equal(System.Math.Pow(2.0, 3.0), Pow(2.0, 3.0));
            Assert.Equal(System.Math.Atan2(1.0, 2.0), Atan2(1.0, 2.0));
            Assert.Equal(System.Math.Abs(-1.5), Abs(-1.5));
        }

        [Fact]
        public void IntegerArguments_StillResolveToSystemMath()
        {
            Assert.Equal(5, Abs(-5));
            Assert.Equal(5L, Abs(-5L));
        }

        [Fact]
        public void DualNumbers_ResolveToTheFacade()
        {
            var (x, y) = DDScalar3<double>.Variables(0.7, 1.3, 0.4) is var v ? (v.x, v.y) : default;

            DDScalar3<double> f = Sin(x) * Sqrt(y);

            Assert.Equal(System.Math.Sin(0.7) * System.Math.Sqrt(1.3), f.Value, precision: 12);
            Assert.Equal(System.Math.Cos(0.7) * System.Math.Sqrt(1.3), f.G(0), precision: 12);
        }
    }
}
