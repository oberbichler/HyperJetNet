using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using Xunit;

namespace HyperJet.Tests
{
    /// <summary>
    /// <c>ToString</c> has to enumerate the coefficients of its own dimension.
    /// </summary>
    /// <remarks>
    /// The generator spelled the format out for two variables, so <c>DDScalar1</c> threw from
    /// <c>G(1)</c> and every dimension above two printed a truncated gradient and Hessian. The
    /// earlier test only covered <c>DDScalar2</c>, which is the one dimension where that was right.
    /// A debugger calls <c>ToString</c> to display a value, so the throwing case broke inspection.
    /// </remarks>
    public class ToStringTests
    {
        public static TheoryData<int> Dimensions
        {
            get
            {
                var data = new TheoryData<int>();
                for (int n = 1; n <= 15; n++) data.Add(n);
                return data;
            }
        }

        private static double[] Point(int n) => Enumerable.Range(0, n).Select(i => 1.0 + i).ToArray();

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void EveryDimension_RendersAllOfItsCoefficients(int n)
        {
            string rendered = RenderGenerated(n);

            // One entry per variable in the gradient, and n rows of n in the Hessian.
            Assert.Equal(n, GradientEntries(rendered));
            Assert.Equal(n, HessianRows(rendered));
        }

        /// <summary>
        /// The dynamic model builds the same text in a loop, so it is an independent reference for
        /// both the format and the coefficients. The expression is chosen to be exactly
        /// representable, which lets the two strings be compared character by character.
        /// </summary>
        [Theory]
        [MemberData(nameof(Dimensions))]
        public void GeneratedStructs_RenderLikeTheDynamicModel(int n)
        {
            double[] point = Point(n);

            DDScalar[] variables = DDScalar.Variables(point, order: 2);
            DDScalar sum = DDScalar.Constant(0.0, n);
            foreach (DDScalar variable in variables) sum += variable;
            DDScalar expected = sum * sum;

            Assert.Equal(expected.ToString(), RenderGenerated(n));
        }

        [Fact]
        public void SingleVariableScalar_DoesNotThrow()
        {
            // The regression that motivated this file: G(1) on a one-variable scalar.
            DDScalar1<double> x = DDScalar1<double>.Variables(1.5);

            string rendered = x.ToString();

            // Formatted through the current culture, so compare against the same rendering rather
            // than a hard-coded "1.5" -- under a German locale the value reads "1,5".
            Assert.StartsWith(1.5.ToString(), rendered);
            Assert.Equal(1, GradientEntries(rendered));
            Assert.Equal(1, HessianRows(rendered));
        }

        #region Rendering the generated struct of a given dimension

        private static string RenderGenerated(int n) =>
            (string)typeof(ToStringTests)
                .GetMethod(nameof(Render), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!.MakeGenericType(typeof(double)))
                .Invoke(null, new object[] { n })!;

        private static string Render<T>(int n) where T : IFloatingPoint<T>
        {
            double[] point = Point(n);

            T sum = T.Zero;
            for (int i = 0; i < n; i++) sum += MakeVariable<T>(i, point[i]);

            return (sum * sum).ToString()!;
        }

        private static T MakeVariable<T>(int index, double value) =>
            (T)typeof(T).GetMethod("Variable", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: new[] { typeof(int), typeof(double) }, modifiers: null)!
                .Invoke(null, new object[] { index, value })!;

        #endregion

        // The rendered form is: "<value> [g: (g0, g1, ...), H: ((h00, ...), (h10, ...))]".
        // Entries are separated by ", "; splitting on that rather than on ',' keeps the parsing
        // correct under a culture that writes decimals as "1,5".

        private static int GradientEntries(string rendered)
        {
            const string opening = "[g: (";
            int start = rendered.IndexOf(opening, StringComparison.Ordinal);
            Assert.True(start >= 0, $"no gradient in: {rendered}");

            int end = rendered.IndexOf("), H: (", start, StringComparison.Ordinal);
            Assert.True(end >= 0, $"no Hessian in: {rendered}");

            return rendered[(start + opening.Length)..end].Split(", ").Length;
        }

        private static int HessianRows(string rendered)
        {
            const string opening = "H: (";
            int start = rendered.IndexOf(opening, StringComparison.Ordinal);
            Assert.True(start >= 0, $"no Hessian in: {rendered}");

            // Everything up to the closing ")]" of the whole rendering; each row opens a bracket.
            return rendered[(start + opening.Length)..^2].Count(c => c == '(');
        }
    }
}
