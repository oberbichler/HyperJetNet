using System;
using System.Numerics;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;

namespace HyperJet.Tests
{
    /// <summary>
    /// <c>T.CreateChecked</c> / <c>CreateSaturating</c> / <c>CreateTruncating</c> are the idiomatic way
    /// to materialise a literal inside generic-math code. Any <see cref="INumberBase{TSelf}"/>
    /// implementation is expected to support at least the built-in numeric types.
    /// </summary>
    public class GenericMathConversionTests
    {
        // CreateXxx are static abstract interface members with a default implementation, so they are
        // only reachable through a constrained type parameter — never off the concrete type name.
        private static T Checked<T, TOther>(TOther value) where T : INumberBase<T> where TOther : INumberBase<TOther>
            => T.CreateChecked(value);

        private static T Saturating<T, TOther>(TOther value) where T : INumberBase<T> where TOther : INumberBase<TOther>
            => T.CreateSaturating(value);

        private static T Truncating<T, TOther>(TOther value) where T : INumberBase<T> where TOther : INumberBase<TOther>
            => T.CreateTruncating(value);

        [Fact]
        public void CreateChecked_FromDouble_ProducesConstant()
        {
            DD2 c = Checked<DD2, double>(0.7);

            Assert.Equal(0.7, c.Value);
            Assert.Equal(0.0, c.G(0));
            Assert.Equal(0.0, c.G(1));
        }

        [Fact]
        public void CreateChecked_FromInt_ProducesConstant()
        {
            DD2 c = Checked<DD2, int>(3);
            Assert.Equal(3.0, c.Value);
        }

        [Fact]
        public void CreateSaturating_FromDouble_ProducesConstant()
        {
            DD2 c = Saturating<DD2, double>(0.7);
            Assert.Equal(0.7, c.Value);
        }

        [Fact]
        public void CreateTruncating_FromDouble_ProducesConstant()
        {
            DD2 c = Truncating<DD2, double>(0.7);
            Assert.Equal(0.7, c.Value);
        }

        [Fact]
        public void GenericAlgorithm_UsingCreateChecked_Works()
        {
            // A textbook generic-math routine: it has no way to write "0.5" other than CreateChecked.
            static T Average<T>(T a, T b) where T : IFloatingPoint<T> =>
                (a + b) / T.CreateChecked(2.0);

            var (x, y) = DD2.Variables(3.0, 6.0);
            DD2 mean = Average(x, y);

            Assert.Equal(4.5, mean.Value);
            Assert.Equal(0.5, mean.G(0));
            Assert.Equal(0.5, mean.G(1));
        }

        [Fact]
        public void ConvertToDouble_ExtractsTheValue()
        {
            var (x, y) = DD2.Variables(3.0, 6.0);
            DD2 f = x * y;

            Assert.True(DD2.TryConvertToChecked(f, out double asDouble));
            Assert.Equal(18.0, asDouble);
        }

        [Fact]
        public void ImplicitConversion_FromUnderlyingType_ProducesConstant()
        {
            DD2 c = 2.5;
            Assert.Equal(2.5, c.Value);
            Assert.Equal(0.0, c.G(0));
        }
    }
}
