using System;
using System.Linq;
using System.Reflection;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// The IEEE 754 remainder is the one function .NET spells two ways.
    /// </summary>
    /// <remarks>
    /// <c>Math.IEEERemainder</c> predates the naming guideline that gave the generic-math surface
    /// <c>Ieee754Remainder</c>, and both survive: the <c>Math</c> class uses the old spelling,
    /// <c>double</c> and <see cref="System.Numerics.IFloatingPointIeee754{TSelf}"/> the new one.
    /// HyperJet reproduces that split rather than picking a side — the facade mirrors <c>Math</c>
    /// and offers both, the types are bound by the interface and offer only <c>Ieee754Remainder</c>.
    /// These tests pin the split down so it stays a decision rather than an accident.
    /// </remarks>
    public class RemainderSpellingTests
    {
        [Theory]
        [InlineData(5.9, 1.0)]
        [InlineData(-5.3, 1.3)]
        [InlineData(17.0, 5.0)]
        [InlineData(0.7, 4.1)]
        public void BothSpellingsComputeTheSameThing(double a0, double b0)
        {
            var (x, y) = DD2.Variables(a0, b0);

            DD2 viaGenericMath = Ieee754Remainder(x, y);
            DD2 viaMathSpelling = IEEERemainder(x, y);

            Assert.Equal(Math.IEEERemainder(a0, b0), viaMathSpelling.Value);
            Assert.Equal(viaGenericMath.Value, viaMathSpelling.Value);
            Assert.Equal(viaGenericMath.G(0), viaMathSpelling.G(0));
            Assert.Equal(viaGenericMath.G(1), viaMathSpelling.G(1));
            Assert.Equal(viaGenericMath.H(0, 1), viaMathSpelling.H(0, 1));
        }

        [Fact]
        public void TheDynamicModelCarriesBothToo()
        {
            var v = DDScalar.Variables(new[] { 5.9, 1.0 });

            Assert.Equal(Ieee754Remainder(v[0], v[1]).Value, IEEERemainder(v[0], v[1]).Value);
            Assert.Equal(Ieee754Remainder(v[0], v[1]).G(1), IEEERemainder(v[0], v[1]).G(1));
        }

        public static TheoryData<int> Dimensions
        {
            get
            {
                var data = new TheoryData<int>();
                for (int n = 1; n <= 15; n++) data.Add(n);
                return data;
            }
        }

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void TheFacadeCarriesBothSpellingsForEveryDimension(int n)
        {
            Type open = Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!;

            var names = typeof(HyperJetMath).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.GetParameters().Any(p =>
                {
                    Type t = p.ParameterType;
                    if (t.IsByRef) t = t.GetElementType()!;
                    return t.IsGenericType && t.GetGenericTypeDefinition() == open;
                }))
                .Select(m => m.Name)
                .ToHashSet();

            Assert.Contains("Ieee754Remainder", names);
            Assert.Contains("IEEERemainder", names);
        }

        /// <summary>
        /// The second spelling stops at the facade. The generated structs implement
        /// <c>IFloatingPointIeee754</c>, which dictates the member name, and a second one beside it
        /// would be duplication rather than fidelity to the platform — <c>double</c> does not carry
        /// <c>IEEERemainder</c> either.
        /// </summary>
        [Theory]
        [MemberData(nameof(Dimensions))]
        public void TheTypesCarryOnlyTheGenericMathSpelling(int n)
        {
            Type closed = Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!.MakeGenericType(typeof(double));

            Assert.NotNull(closed.GetMethod("Ieee754Remainder"));
            Assert.Null(closed.GetMethod("IEEERemainder"));
        }

        [Fact]
        public void DotNetItselfCarriesTheSplit()
        {
            // The premise this design rests on, asserted rather than assumed.
            Assert.NotNull(typeof(Math).GetMethod("IEEERemainder", new[] { typeof(double), typeof(double) }));
            Assert.Null(typeof(Math).GetMethod("Ieee754Remainder", new[] { typeof(double), typeof(double) }));

            Assert.NotNull(typeof(double).GetMethod("Ieee754Remainder", new[] { typeof(double), typeof(double) }));
            Assert.Null(typeof(double).GetMethod("IEEERemainder", new[] { typeof(double), typeof(double) }));
        }
    }
}
