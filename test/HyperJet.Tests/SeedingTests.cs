using System;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Seeding a <see cref="DDScalar"/> now happens at construction. <c>SetG</c> used to be the only
    /// way to express a tangent other than a unit basis vector; <see cref="DDScalar.WithGradient"/>
    /// takes over that job without handing out a mutable handle on a shared buffer.
    /// </summary>
    public class SeedingTests
    {
        [Fact]
        public void WithGradient_SeedsTheGivenTangent()
        {
            double[] tangent = { 0.4, -0.9, 2.5 };

            DDScalar x = DDScalar.WithGradient(1.7, tangent);

            Assert.Equal(1.7, x.Value);
            Assert.Equal(3, x.Size);
            Assert.Equal(2, x.Order);
            Assert.Equal(tangent, x.GetGradient());

            // Second order is present and seeded flat, as it is for Variable.
            Assert.Equal(0.0, x.H(0, 0));
            Assert.Equal(0.0, x.H(1, 2));
        }

        [Fact]
        public void WithGradient_HonoursTheRequestedOrder()
        {
            DDScalar x = DDScalar.WithGradient(1.0, new[] { 1.0, 2.0 }, order: 1);

            Assert.Equal(1, x.Order);
            Assert.Equal(Kernel.GetDataLength(2, 1), x.DataLength);
            Assert.Throws<InvalidOperationException>(() => x.H(0, 0));
        }

        [Fact]
        public void WithGradient_ReproducesVariable()
        {
            DDScalar viaFactory = DDScalar.WithGradient(2.5, new[] { 0.0, 1.0, 0.0 });
            DDScalar viaVariable = DDScalar.Variable(1, 2.5, size: 3);

            Assert.Equal(viaVariable.AsReadOnlySpan().ToArray(), viaFactory.AsReadOnlySpan().ToArray());
        }

        /// <summary>
        /// The reason an arbitrary tangent is worth having: one derivative slot carries the whole
        /// directional derivative, so the cost does not grow with the number of inputs.
        /// </summary>
        [Fact]
        public void ASingleSlotCarriesADirectionalDerivative()
        {
            const double x0 = 0.7, y0 = 1.3;
            double[] direction = { 0.4, -0.9 };

            // One slot, seeded with the direction component of each input.
            DDScalar xd = DDScalar.WithGradient(x0, new[] { direction[0] }, order: 1);
            DDScalar yd = DDScalar.WithGradient(y0, new[] { direction[1] }, order: 1);
            DDScalar along = Sin(xd) * Exp(yd);

            // The full gradient, for comparison.
            var v = DDScalar.Variables(new[] { x0, y0 }, order: 1);
            DDScalar full = Sin(v[0]) * Exp(v[1]);
            double expected = full.G(0) * direction[0] + full.G(1) * direction[1];

            Assert.Equal(expected, along.G(0), precision: 12);
            Assert.Equal(1, along.Size);
        }

        [Fact]
        public void WithGradient_RejectsAnUnsupportedOrder()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => DDScalar.WithGradient(1.0, new[] { 1.0 }, order: 3));
        }

        [Fact]
        public void WithGradient_AcceptsAnEmptyTangent()
        {
            DDScalar constant = DDScalar.WithGradient(4.0, ReadOnlySpan<double>.Empty);

            Assert.Equal(0, constant.Size);
            Assert.Equal(4.0, constant.Value);
        }

        /// <summary>The seeded scalar owns its buffer; the caller's array is not retained.</summary>
        [Fact]
        public void WithGradient_CopiesTheTangent()
        {
            double[] tangent = { 1.0, 2.0 };

            DDScalar x = DDScalar.WithGradient(0.0, tangent);
            tangent[0] = 99.0;

            Assert.Equal(1.0, x.G(0));
        }
    }
}
