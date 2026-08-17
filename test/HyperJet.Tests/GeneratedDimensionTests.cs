using System;
using System.Numerics;
using System.Reflection;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// Evaluates one non-trivial expression in every generated dimension
    /// (<c>DDScalar1</c>..<c>DDScalar15</c>) and compares value, gradient and Hessian against the
    /// dynamic <see cref="DDScalar"/> model computing the structurally identical expression.
    /// </summary>
    /// <remarks>
    /// The generated structs have data lengths 3..136, so the sweep also covers all three SIMD
    /// widths inside the generated code paths. <c>DDScalar1</c>..<c>DDScalar15</c> are distinct
    /// closed types with no common non-generic base, so the accessors are reached by reflection
    /// while the arithmetic itself runs through the generic-math interfaces.
    /// </remarks>
    public class GeneratedDimensionTests
    {
        /// <summary>
        /// The expression under test, written once against .NET generic math.
        /// Mixes products, transcendentals and a division so that every kernel arity contributes.
        /// </summary>
        /// <remarks>
        /// Constants arrive through the <paramref name="constant"/> factory rather than
        /// <c>T.CreateChecked</c>, which currently throws for these types — see
        /// <see cref="GenericMathConversionTests"/>.
        /// </remarks>
        private static T Expression<T>(T[] v, Func<double, T> constant)
            where T : IFloatingPoint<T>, ITrigonometricFunctions<T>, IExponentialFunctions<T>,
                      ILogarithmicFunctions<T>, IRootFunctions<T>, IHyperbolicFunctions<T>
        {
            int n = v.Length;

            T weighted = T.Zero;
            T coupled = T.Zero;
            for (int i = 0; i < n; i++)
            {
                weighted += constant(0.7 + 0.1 * i) * v[i];
                coupled += v[i] * v[(i + 1) % n];
            }

            T four = constant(4.0);
            T three = constant(3.0);
            T third = constant(0.3);

            return T.Sin(weighted) * T.Exp(third * coupled)
                 + T.Sqrt(four + weighted * weighted)
                 - T.Log(three + coupled * coupled)
                 + T.Tanh(weighted)
                 + weighted / (three + coupled * coupled);
        }

        /// <summary>The same expression for the dynamic model, which is outside generic math.</summary>
        private static DDScalar Expression(DDScalar[] v)
        {
            int n = v.Length;

            DDScalar weighted = DDScalar.Constant(0.0, n);
            DDScalar coupled = DDScalar.Constant(0.0, n);
            for (int i = 0; i < n; i++)
            {
                weighted += (0.7 + 0.1 * i) * v[i];
                coupled += v[i] * v[(i + 1) % n];
            }

            return Sin(weighted) * Exp(0.3 * coupled)
                 + Sqrt(4.0 + weighted * weighted)
                 - Log(3.0 + coupled * coupled)
                 + Tanh(weighted)
                 + weighted / (3.0 + coupled * coupled);
        }

        private static double[] Point(int n)
        {
            double[] values = new double[n];
            for (int i = 0; i < n; i++) values[i] = 0.3 + 0.11 * i - 0.02 * i * i;
            return values;
        }

        [Fact] public void Dimension01_MatchesDynamicModel() => Check<DDScalar1<double>>(1);
        [Fact] public void Dimension02_MatchesDynamicModel() => Check<DDScalar2<double>>(2);
        [Fact] public void Dimension03_MatchesDynamicModel() => Check<DDScalar3<double>>(3);
        [Fact] public void Dimension04_MatchesDynamicModel() => Check<DDScalar4<double>>(4);
        [Fact] public void Dimension05_MatchesDynamicModel() => Check<DDScalar5<double>>(5);
        [Fact] public void Dimension06_MatchesDynamicModel() => Check<DDScalar6<double>>(6);
        [Fact] public void Dimension07_MatchesDynamicModel() => Check<DDScalar7<double>>(7);
        [Fact] public void Dimension08_MatchesDynamicModel() => Check<DDScalar8<double>>(8);
        [Fact] public void Dimension09_MatchesDynamicModel() => Check<DDScalar9<double>>(9);
        [Fact] public void Dimension10_MatchesDynamicModel() => Check<DDScalar10<double>>(10);
        [Fact] public void Dimension11_MatchesDynamicModel() => Check<DDScalar11<double>>(11);
        [Fact] public void Dimension12_MatchesDynamicModel() => Check<DDScalar12<double>>(12);
        [Fact] public void Dimension13_MatchesDynamicModel() => Check<DDScalar13<double>>(13);
        [Fact] public void Dimension14_MatchesDynamicModel() => Check<DDScalar14<double>>(14);
        [Fact] public void Dimension15_MatchesDynamicModel() => Check<DDScalar15<double>>(15);

        private static void Check<T>(int n)
            where T : IFloatingPoint<T>, ITrigonometricFunctions<T>, IExponentialFunctions<T>,
                      ILogarithmicFunctions<T>, IRootFunctions<T>, IHyperbolicFunctions<T>
        {
            double[] point = Point(n);

            T[] staticVars = new T[n];
            for (int i = 0; i < n; i++) staticVars[i] = MakeVariable<T>(i, point[i]);
            T staticResult = Expression(staticVars, MakeConstant<T>);

            DDScalar[] dynamicVars = DDScalar.Variables(point, order: 2);
            DDScalar dynamicResult = Expression(dynamicVars);

            Close($"n={n} value", dynamicResult.Value, ValueOf(staticResult));

            for (int i = 0; i < n; i++)
            {
                Close($"n={n} G({i})", dynamicResult.G(i), GradientOf(staticResult, i));
                for (int j = 0; j < n; j++)
                {
                    Close($"n={n} H({i},{j})", dynamicResult.H(i, j), HessianOf(staticResult, i, j));
                    // The triangular Hessian storage must be symmetric under index swap.
                    Assert.Equal(HessianOf(staticResult, i, j), HessianOf(staticResult, j, i));
                }
            }
        }

        #region Reflection accessors

        private static T MakeVariable<T>(int index, double value)
        {
            MethodInfo method = typeof(T).GetMethod("Variable", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: new[] { typeof(int), typeof(double) }, modifiers: null)
                ?? throw new InvalidOperationException($"{typeof(T)} has no static Variable(int, double).");
            return (T)method.Invoke(null, new object[] { index, value })!;
        }

        private static T MakeConstant<T>(double value)
        {
            MethodInfo method = typeof(T).GetMethod("Constant", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: new[] { typeof(double) }, modifiers: null)
                ?? throw new InvalidOperationException($"{typeof(T)} has no static Constant(double).");
            return (T)method.Invoke(null, new object[] { value })!;
        }

        private static double ValueOf<T>(T scalar) =>
            (double)typeof(T).GetProperty("Value")!.GetValue(scalar)!;

        private static double GradientOf<T>(T scalar, int i) =>
            (double)typeof(T).GetMethod("G")!.Invoke(scalar, new object[] { i })!;

        private static double HessianOf<T>(T scalar, int i, int j) =>
            (double)typeof(T).GetMethod("H")!.Invoke(scalar, new object[] { i, j })!;

        #endregion

        private static void Close(string what, double expected, double actual)
        {
            double tolerance = 1e-10 * (1.0 + Math.Abs(expected));
            Assert.True(Math.Abs(expected - actual) <= tolerance,
                $"{what}: dynamic model gave {expected:R}, generated struct gave {actual:R}");
        }
    }
}
