using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using DD2 = HyperJet.DDScalar2<double>;

namespace HyperJet.Tests
{
    /// <summary>
    /// The three computational models must expose the same mathematical vocabulary. This test walks
    /// the public surface by reflection so a function added to one model — or forgotten in another —
    /// is caught mechanically rather than by reading three files side by side.
    /// </summary>
    public class FunctionParityTests
    {
        /// <summary>
        /// The reference vocabulary: every mathematical function the generated structs provide
        /// through .NET generic math. Non-mathematical members (parsing, formatting, comparison,
        /// classification, rounding, factories) are out of scope.
        /// </summary>
        private static readonly string[] ExpectedFunctions =
        {
            "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Atan2", "SinCos",
            "SinPi", "CosPi", "TanPi", "AsinPi", "AcosPi", "AtanPi", "Atan2Pi", "SinCosPi",
            "Exp", "Exp2", "Exp10", "ExpM1",
            "Log", "Log2", "Log10", "LogP1",
            "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",
            "Pow", "Sqrt", "Cbrt", "RootN", "Hypot", "Abs",
        };

        private static HashSet<string> StaticNames(Type type, Type argument) =>
            type.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.GetParameters().Any(p => p.ParameterType == argument || p.ParameterType == argument.MakeByRefType()))
                .Select(m => m.Name)
                .ToHashSet();

        private static HashSet<string> InstanceNames(Type type) =>
            type.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Select(m => m.Name)
                .ToHashSet();

        /// <summary>The closed type <c>DDScalar{n}&lt;double&gt;</c> for every generated dimension.</summary>
        private static Type ClosedStruct(int n) =>
            Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!.MakeGenericType(typeof(double));

        /// <summary>
        /// The facade overloads are generic (<c>Sin&lt;T&gt;(in DDScalar{n}&lt;T&gt;)</c>), so their
        /// parameter type is the open <c>DDScalar{n}&lt;T&gt;</c>. Match on the generic definition.
        /// </summary>
        private static HashSet<string> FacadeNamesForDimension(int n)
        {
            Type open = Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!;

            return typeof(HyperJetMath).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.GetParameters().Any(p =>
                {
                    Type t = p.ParameterType;
                    if (t.IsByRef) t = t.GetElementType()!;
                    return t.IsGenericType && t.GetGenericTypeDefinition() == open;
                }))
                .Select(m => m.Name)
                .ToHashSet();
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
        public void GeneratedStruct_ProvidesTheFullVocabulary(int n)
        {
            Type closed = ClosedStruct(n);
            AssertCovers($"DDScalar{n}<double>", StaticNames(closed, closed));
        }

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void HyperJetMathFacade_ProvidesTheFullVocabularyForEveryDimension(int n)
        {
            AssertCovers($"HyperJetMath (DDScalar{n})", FacadeNamesForDimension(n));
        }

        [Fact]
        public void HyperJetMathFacade_ProvidesTheFullVocabularyForDDScalar()
        {
            AssertCovers("HyperJetMath (DDScalar)", StaticNames(typeof(HyperJetMath), typeof(DDScalar)));
        }

        [Fact]
        public void DDScalarSpan_ProvidesTheFullVocabulary()
        {
            AssertCovers("DDScalarSpan", InstanceNames(typeof(DDScalarSpan)));
        }

        /// <summary>
        /// Members that are not generic-math functions but still have to exist on every model.
        /// <c>Evaluate</c> is here because it was already lost once: it existed in the T4-generated
        /// code and disappeared in the refactoring without anything noticing.
        /// </summary>
        [Fact]
        public void EveryModel_ExposesTheInstanceLevelMembers()
        {
            string[] expected = { "Evaluate", "G", "H", "GetGradient", "GetHessian", "AsSpan", "AsReadOnlySpan" };

            var surfaces = new List<(string Name, HashSet<string> Members)>
            {
                ("DDScalar", InstanceNames(typeof(DDScalar))),
                ("DDScalarSpan", InstanceNames(typeof(DDScalarSpan))),
            };

            for (int n = 1; n <= 15; n++)
            {
                surfaces.Add(($"DDScalar{n}<double>", InstanceNames(ClosedStruct(n))));
            }

            foreach (var (name, members) in surfaces)
            {
                string[] missing = expected.Where(m => !members.Contains(m)).ToArray();
                Assert.True(missing.Length == 0, $"{name} is missing: {string.Join(", ", missing)}");
            }
        }

        private static void AssertCovers(string model, HashSet<string> available)
        {
            string[] missing = ExpectedFunctions.Where(name => !available.Contains(name)).ToArray();

            Assert.True(missing.Length == 0, $"{model} is missing: {string.Join(", ", missing)}");
        }

        /// <summary>
        /// Spot-check that the three models do not merely share names but agree numerically,
        /// including in their derivatives, for the functions that were added last.
        /// </summary>
        [Fact]
        public void AllThreeModels_AgreeOnTheNewlyAddedFunctions()
        {
            const int size = 2, order = 2;
            const double x0 = 0.31, y0 = 0.24;

            var (sx, sy) = DD2.Variables(x0, y0);
            var (dx, dy) = DDScalar.Variables(new[] { x0, y0 }, order);

            // A value in (0, 1) so every domain below is satisfied.
            DD2 argStatic = 0.6 * sx + 0.7 * sy + 0.4 * sx * sy;
            DDScalar argDynamic = 0.6 * dx + 0.7 * dy + 0.4 * dx * dy;

            var dynamicOps = new Dictionary<string, Func<DDScalar, DDScalar>>
            {
                ["SinPi"] = a => HyperJetMath.SinPi(a),
                ["CosPi"] = a => HyperJetMath.CosPi(a),
                ["TanPi"] = a => HyperJetMath.TanPi(a),
                ["AsinPi"] = a => HyperJetMath.AsinPi(a),
                ["AcosPi"] = a => HyperJetMath.AcosPi(a),
                ["AtanPi"] = a => HyperJetMath.AtanPi(a),
                ["Exp2"] = a => HyperJetMath.Exp2(a),
                ["Exp10"] = a => HyperJetMath.Exp10(a),
                ["ExpM1"] = a => HyperJetMath.ExpM1(a),
                ["LogP1"] = a => HyperJetMath.LogP1(a),
                ["Asinh"] = a => HyperJetMath.Asinh(a),
                ["Atanh"] = a => HyperJetMath.Atanh(a),
                ["RootN"] = a => HyperJetMath.RootN(a, 3),
            };

            var staticOps = new Dictionary<string, Func<DD2, DD2>>
            {
                ["SinPi"] = DD2.SinPi,
                ["CosPi"] = DD2.CosPi,
                ["TanPi"] = DD2.TanPi,
                ["AsinPi"] = DD2.AsinPi,
                ["AcosPi"] = DD2.AcosPi,
                ["AtanPi"] = DD2.AtanPi,
                ["Exp2"] = DD2.Exp2,
                ["Exp10"] = DD2.Exp10,
                ["ExpM1"] = DD2.ExpM1,
                ["LogP1"] = DD2.LogP1,
                ["Asinh"] = DD2.Asinh,
                ["Atanh"] = DD2.Atanh,
                ["RootN"] = a => DD2.RootN(a, 3),
            };

            foreach (string name in staticOps.Keys)
            {
                DD2 fromStatic = staticOps[name](argStatic);
                DDScalar fromDynamic = dynamicOps[name](argDynamic);

                Close($"{name} value", fromStatic.Value, fromDynamic.Value);
                Close($"{name} G(0)", fromStatic.G(0), fromDynamic.G(0));
                Close($"{name} G(1)", fromStatic.G(1), fromDynamic.G(1));
                Close($"{name} H(0,0)", fromStatic.H(0, 0), fromDynamic.H(0, 0));
                Close($"{name} H(0,1)", fromStatic.H(0, 1), fromDynamic.H(0, 1));
                Close($"{name} H(1,1)", fromStatic.H(1, 1), fromDynamic.H(1, 1));
            }

            // The span model runs the same list through its destination-based methods.
            int n = Kernel.GetDataLength(size, order);
            Span<double> destBuffer = stackalloc double[n];
            var arg = new DDScalarSpan(argDynamic.AsSpan(), size, order);
            var dest = new DDScalarSpan(destBuffer, size, order);

            arg.SinPi(dest); CompareSpan("SinPi", staticOps["SinPi"](argStatic), dest);
            arg.CosPi(dest); CompareSpan("CosPi", staticOps["CosPi"](argStatic), dest);
            arg.TanPi(dest); CompareSpan("TanPi", staticOps["TanPi"](argStatic), dest);
            arg.AsinPi(dest); CompareSpan("AsinPi", staticOps["AsinPi"](argStatic), dest);
            arg.AcosPi(dest); CompareSpan("AcosPi", staticOps["AcosPi"](argStatic), dest);
            arg.AtanPi(dest); CompareSpan("AtanPi", staticOps["AtanPi"](argStatic), dest);
            arg.Exp2(dest); CompareSpan("Exp2", staticOps["Exp2"](argStatic), dest);
            arg.Exp10(dest); CompareSpan("Exp10", staticOps["Exp10"](argStatic), dest);
            arg.ExpM1(dest); CompareSpan("ExpM1", staticOps["ExpM1"](argStatic), dest);
            arg.LogP1(dest); CompareSpan("LogP1", staticOps["LogP1"](argStatic), dest);
            arg.Asinh(dest); CompareSpan("Asinh", staticOps["Asinh"](argStatic), dest);
            arg.Atanh(dest); CompareSpan("Atanh", staticOps["Atanh"](argStatic), dest);
            arg.RootN(3, dest); CompareSpan("RootN", staticOps["RootN"](argStatic), dest);
        }

        private static void CompareSpan(string name, in DD2 expected, in DDScalarSpan actual)
        {
            Close($"span {name} value", expected.Value, actual.Value);
            Close($"span {name} G(0)", expected.G(0), actual.G(0));
            Close($"span {name} G(1)", expected.G(1), actual.G(1));
            Close($"span {name} H(0,0)", expected.H(0, 0), actual.H(0, 0));
            Close($"span {name} H(0,1)", expected.H(0, 1), actual.H(0, 1));
            Close($"span {name} H(1,1)", expected.H(1, 1), actual.H(1, 1));
        }

        private static void Close(string what, double expected, double actual)
        {
            double tolerance = 1e-12 * (1.0 + Math.Abs(expected));
            Assert.True(Math.Abs(expected - actual) <= tolerance,
                $"{what}: {expected:R} vs {actual:R}");
        }
    }
}
