using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using Xunit;
using static HyperJet.HyperJetMath;

namespace HyperJet.Tests
{
    /// <summary>
    /// <c>DScalar1</c>..<c>DScalar15</c> carry a value and a gradient but no Hessian.
    /// </summary>
    /// <remarks>
    /// They come from the same generator template as the second-order family, so the derivative
    /// formulas exist once. What these tests establish is that the order-1 path through that
    /// template is wired up correctly: the storage is the smaller one, the first derivatives agree
    /// with what the second-order family computes, and no Hessian is exposed or paid for.
    /// </remarks>
    public class FirstOrderScalarTests
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

        #region Shape

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void StoresOnlyTheValueAndTheGradient(int n)
        {
            Type type = FirstOrder(n);

            Assert.Equal(n, (int)type.GetField("Size")!.GetRawConstantValue()!);
            Assert.Equal(1, (int)type.GetField("Order")!.GetRawConstantValue()!);
            Assert.Equal(1 + n, (int)type.GetField("DataLength")!.GetRawConstantValue()!);

            // Which is what the kernel expects for a first-order scalar of this size.
            Assert.Equal(Kernel.GetDataLength(n, order: 1), (int)type.GetField("DataLength")!.GetRawConstantValue()!);
        }

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void ExposesNoHessian(int n)
        {
            Type type = FirstOrder(n);

            Assert.Null(type.GetMethod("H"));
            Assert.Null(type.GetMethod("SetH"));
            Assert.Null(type.GetMethod("GetHessian"));
        }

        /// <summary>
        /// The point of the type: at fifteen variables a first-order scalar is 16 doubles where the
        /// second-order one is 136.
        /// </summary>
        [Theory]
        [MemberData(nameof(Dimensions))]
        public void IsSmallerThanTheSecondOrderScalar(int n)
        {
            int first = (int)FirstOrder(n).GetField("DataLength")!.GetRawConstantValue()!;
            int second = (int)SecondOrder(n).GetField("DataLength")!.GetRawConstantValue()!;

            Assert.Equal(1 + n, first);
            Assert.Equal((n + 1) * (n + 2) / 2, second);
            Assert.True(first <= second);
        }

        #endregion

        #region The gradients agree with the second-order family

        /// <summary>
        /// Both families run the same formulas through the same kernel, so for every function the
        /// first derivatives must come out identical -- bit for bit, not merely close.
        /// </summary>
        [Theory]
        [MemberData(nameof(Dimensions))]
        public void EveryFunction_MatchesTheSecondOrderGradient(int n)
        {
            double[] point = Point(n);

            double[] first = Invoke(nameof(EvaluateAll), FirstOrder(n), point);
            double[] second = Invoke(nameof(EvaluateAll), SecondOrder(n), point);

            Assert.Equal(second, first);
        }

        /// <summary>Value and gradient of a long chain of every generic-math function.</summary>
        private static double[] EvaluateAll<T>(double[] point)
            where T : IFloatingPointIeee754<T>
        {
            int n = point.Length;

            T[] v = new T[n];
            for (int i = 0; i < n; i++) v[i] = MakeVariable<T>(i, point[i]);

            T two = T.One + T.One;
            T half = T.One / two;

            // Kept inside the domains every function below needs.
            T a = T.Zero;
            for (int i = 0; i < n; i++) a += v[i];
            a = half * T.Tanh(a);                       // in (-0.5, 0.5)
            T positive = T.One + half + a;              // around 1.5

            T result = T.Sin(a) + T.Cos(a) + T.Tan(a) + T.Asin(a) + T.Acos(a) + T.Atan(a)
                     + T.SinPi(a) + T.CosPi(a) + T.AsinPi(a) + T.AcosPi(a) + T.AtanPi(a)
                     + T.Exp(a) + T.Exp2(a) + T.Exp10(a) + T.ExpM1(a)
                     + T.Log(positive) + T.Log2(positive) + T.Log10(positive) + T.LogP1(positive)
                     + T.Sinh(a) + T.Cosh(a) + T.Tanh(a) + T.Asinh(a) + T.Acosh(positive) + T.Atanh(a)
                     + T.Sqrt(positive) + T.Cbrt(positive) + T.RootN(positive, 5)
                     + T.Pow(positive, a) + T.Hypot(a, positive) + T.Atan2(a, positive)
                     + T.FusedMultiplyAdd(a, positive, a) + T.Ieee754Remainder(positive, T.One)
                     + T.Abs(a) + a * positive - a / positive + T.ScaleB(a, 3);

            double[] coefficients = new double[1 + n];
            coefficients[0] = ToDouble(Value(result));
            for (int i = 0; i < n; i++) coefficients[1 + i] = ToDouble(Gradient(result, i));

            return coefficients;
        }

        #endregion

        #region Evaluate degenerates to the linear model

        [Theory]
        [MemberData(nameof(Dimensions))]
        public void Evaluate_IsTheLinearModel(int n)
        {
            double[] point = Point(n);
            double[] rendered = Invoke(nameof(EvaluateTaylor), FirstOrder(n), point);

            // value, then the model at the offset, then the gradient
            double value = rendered[0];
            double model = rendered[1];
            double expected = value;
            for (int i = 0; i < n; i++) expected += rendered[2 + i] * Offset(i);

            Assert.Equal(expected, model, precision: 12);
        }

        private static double Offset(int i) => 0.1 - 0.03 * i;

        private static double[] EvaluateTaylor<T>(double[] point) where T : IFloatingPointIeee754<T>
        {
            int n = point.Length;

            T[] v = new T[n];
            for (int i = 0; i < n; i++) v[i] = MakeVariable<T>(i, point[i]);

            T sum = T.Zero;
            for (int i = 0; i < n; i++) sum += v[i];
            T f = T.Sin(sum) * T.Exp(sum);

            double[] offsets = new double[n];
            for (int i = 0; i < n; i++) offsets[i] = Offset(i);

            double[] result = new double[2 + n];
            result[0] = ToDouble(Value(f));
            result[1] = EvaluateAt(f, offsets);
            for (int i = 0; i < n; i++) result[2 + i] = ToDouble(Gradient(f, i));

            return result;
        }

        #endregion

        #region The facade and the aliases reach the new family

        [Fact]
        public void TheFacadeServesTheFirstOrderFamily()
        {
            // Compile-time proof that HyperJetMath carries overloads for DScalar{n} too.
            var (x, y) = DScalar2<double>.Variables(0.7, 1.3);

            DScalar2<double> f = Sin(x) * Exp(y) + Hypot(x, y);

            var (sx, sy) = DDScalar2<double>.Variables(0.7, 1.3);
            DDScalar2<double> expected = Sin(sx) * Exp(sy) + Hypot(sx, sy);

            Assert.Equal(expected.Value, f.Value);
            Assert.Equal(expected.G(0), f.G(0));
            Assert.Equal(expected.G(1), f.G(1));
        }

        [Fact]
        public void SeedsVariablesLikeTheSecondOrderFamily()
        {
            // The unqualified DScalar3 alias is a consumer-side convenience from
            // build/HyperJet.targets, which the packaging job checks; here the type is explicit.
            DScalar3<double> x = DScalar3<double>.Variable(0, 2.0);

            Assert.Equal(2.0, x.Value);
            Assert.Equal(1.0, x.G(0));
            Assert.Equal(0.0, x.G(2));
        }

        [Fact]
        public void RendersWithoutAHessian()
        {
            var (x, y) = DScalar2<double>.Variables(1.0, 2.0);

            string rendered = (x * y).ToString();

            Assert.Contains("g:", rendered);
            Assert.DoesNotContain("H:", rendered);
        }

        [Fact]
        public void FirstOrderArithmetic_DoesNotAllocate()
        {
            var (x, y, _, _) = DScalar4<double>.Variables(1.3, 2.7, 0.5, 1.1);
            long[] measured = new long[10];

            for (int attempt = 0; attempt < measured.Length; attempt++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();

                for (int i = 0; i < 1000; i++) _sink += ((x * y) / (x - y) + DScalar4<double>.Sin(x)).Value;

                measured[attempt] = GC.GetAllocatedBytesForCurrentThread() - before;
                if (measured[attempt] == 0) return;
            }

            Assert.Fail($"first-order arithmetic kept allocating: {string.Join(", ", measured)} bytes");
        }

        private static double _sink;

        #endregion

        #region Reflection plumbing

        private static Type FirstOrder(int n) =>
            Type.GetType($"HyperJet.DScalar{n}`1, HyperJet")!.MakeGenericType(typeof(double));

        private static Type SecondOrder(int n) =>
            Type.GetType($"HyperJet.DDScalar{n}`1, HyperJet")!.MakeGenericType(typeof(double));

        private static double[] Point(int n) =>
            Enumerable.Range(0, n).Select(i => 0.4 - 0.13 * i + 0.02 * i * i).ToArray();

        private static double[] Invoke(string method, Type scalar, double[] point) =>
            (double[])typeof(FirstOrderScalarTests)
                .GetMethod(method, BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(scalar)
                .Invoke(null, new object[] { point })!;

        private static T MakeVariable<T>(int index, double value) =>
            (T)typeof(T).GetMethod("Variable", BindingFlags.Public | BindingFlags.Static,
                binder: null, types: new[] { typeof(int), typeof(double) }, modifiers: null)!
                .Invoke(null, new object[] { index, value })!;

        private static object Value<T>(T scalar) => typeof(T).GetProperty("Value")!.GetValue(scalar)!;

        private static object Gradient<T>(T scalar, int i) =>
            typeof(T).GetMethod("G")!.Invoke(scalar, new object[] { i })!;

        /// <summary>
        /// Goes through the fixed-arity overload: the span one takes a <c>ReadOnlySpan</c>, which is
        /// a ref struct and cannot travel through reflection. Its parameters are the coefficient
        /// type -- double here -- not the scalar type.
        /// </summary>
        private static double EvaluateAt<T>(T scalar, double[] offsets) =>
            (double)typeof(T).GetMethod("Evaluate", offsets.Select(_ => typeof(double)).ToArray())!
                .Invoke(scalar, offsets.Cast<object>().ToArray())!;

        private static double ToDouble(object value) => (double)value;

        #endregion
    }
}
