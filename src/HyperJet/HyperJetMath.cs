using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace HyperJet
{
    /// <summary>
    /// Provides mathematical operations for the dynamic <see cref="DDScalar"/> in the free-function
    /// spelling of <see cref="System.Math"/>, so that <c>using static HyperJet.HyperJetMath;</c> lets
    /// an expression be written the same way for every computational model.
    /// </summary>
    /// <remarks>
    /// The matching overloads for the generated <c>DDScalar1</c>..<c>DDScalar15</c> structs live in the
    /// other half of this partial class, which the source generator emits. Those forward to the
    /// structs' own generic-math members rather than repeating the derivative formulas.
    /// </remarks>
    public static partial class HyperJetMath
    {
        #region DDScalar (Dynamic) Functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Sin(in DDScalar a)
        {
            double f = Math.Sin(a.Value);
            double da = Math.Cos(a.Value);
            double daa = -f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Cos(in DDScalar a)
        {
            double f = Math.Cos(a.Value);
            double da = -Math.Sin(a.Value);
            double daa = -f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Tan(in DDScalar a)
        {
            double f = Math.Tan(a.Value);
            double da = f * f + 1.0;
            double daa = da * 2.0 * f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Asin(in DDScalar a)
        {
            double f = Math.Asin(a.Value);
            double tmp = 1.0 - a.Value * a.Value;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = da * a.Value / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Acos(in DDScalar a)
        {
            double f = Math.Acos(a.Value);
            double tmp = 1.0 - a.Value * a.Value;
            double da = -1.0 / Math.Sqrt(tmp);
            double daa = da * a.Value / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Atan(in DDScalar a)
        {
            double f = Math.Atan(a.Value);
            double da = 1.0 / (a.Value * a.Value + 1.0);
            double daa = -da * da * 2.0 * a.Value;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Atan2(in DDScalar y, in DDScalar x)
        {
            if (y.Size != x.Size || y.Order != x.Order)
                throw new InvalidOperationException("Incompatible sizes or orders for Atan2.");

            double tmp = y.Value * y.Value + x.Value * x.Value;
            double f = Math.Atan2(y.Value, x.Value);
            double da = x.Value / tmp;
            double db = -y.Value / tmp;
            double daa = db * da * 2.0;
            double dab = db * db - da * da;
            double dbb = -daa;

            DDScalar result = new DDScalar(y.Size, y.Order);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                y.AsReadOnlySpan(), x.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), y.Size, y.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Exp(in DDScalar a)
        {
            double f = Math.Exp(a.Value);
            double da = f;
            double daa = f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Log(in DDScalar a)
        {
            double f = Math.Log(a.Value);
            double da = 1.0 / a.Value;
            double daa = -da * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Log10(in DDScalar a)
        {
            double f = Math.Log10(a.Value);
            double ln10 = Math.Log(10.0);
            double da = 1.0 / (a.Value * ln10);
            double daa = -da / a.Value;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Sqrt(in DDScalar a)
        {
            double f = Math.Sqrt(a.Value);
            double da = 1.0 / (2.0 * f);
            double daa = -da / (2.0 * a.Value);

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Pow(in DDScalar a, double b)
        {
            double f = Math.Pow(a.Value, b);
            double da = b * Math.Pow(a.Value, b - 1.0);
            double daa = (b - 1.0) * b * Math.Pow(a.Value, b - 2.0);

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Hypot(in DDScalar a, in DDScalar b)
        {
            if (a.Size != b.Size || a.Order != b.Order)
                throw new InvalidOperationException("Incompatible sizes or orders for Hypot.");

            double f = Math.Sqrt(a.Value * a.Value + b.Value * b.Value);
            double f3 = f * f * f;
            double da = a.Value / f;
            double db = b.Value / f;
            double daa = b.Value * b.Value / f3;
            double dab = -a.Value * b.Value / f3;
            double dbb = a.Value * a.Value / f3;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Abs(in DDScalar a)
        {
            return a.Value < 0 ? -a : a;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Sinh(in DDScalar a)
        {
            double f = Math.Sinh(a.Value);
            double da = Math.Cosh(a.Value);
            double daa = f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Cosh(in DDScalar a)
        {
            double f = Math.Cosh(a.Value);
            double da = Math.Sinh(a.Value);
            double daa = f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Tanh(in DDScalar a)
        {
            double f = Math.Tanh(a.Value);
            double da = 1.0 - f * f;
            double daa = -2.0 * f * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Cbrt(in DDScalar a)
        {
            double f = Math.Cbrt(a.Value);
            double da = 1.0 / (3.0 * f * f);
            double daa = -2.0 * da / (3.0 * a.Value);

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Log2(in DDScalar a)
        {
            double f = Math.Log2(a.Value);
            double ln2 = Math.Log(2.0);
            double da = 1.0 / (a.Value * ln2);
            double daa = -da / a.Value;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar SinPi(in DDScalar a)
        {
            double f = Math.Sin(Math.PI * a.Value);
            double da = Math.PI * Math.Cos(Math.PI * a.Value);
            double daa = -Math.PI * Math.PI * f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar CosPi(in DDScalar a)
        {
            double f = Math.Cos(Math.PI * a.Value);
            double da = -Math.PI * Math.Sin(Math.PI * a.Value);
            double daa = -Math.PI * Math.PI * f;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar TanPi(in DDScalar a)
        {
            double f = Math.Tan(Math.PI * a.Value);
            double da = Math.PI * (f * f + 1.0);
            double daa = 2.0 * Math.PI * f * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar AsinPi(in DDScalar a)
        {
            double f = Math.Asin(a.Value) / Math.PI;
            double tmp = 1.0 - a.Value * a.Value;
            double da = 1.0 / (Math.PI * Math.Sqrt(tmp));
            double daa = a.Value * da / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar AcosPi(in DDScalar a)
        {
            double f = Math.Acos(a.Value) / Math.PI;
            double tmp = 1.0 - a.Value * a.Value;
            double da = -1.0 / (Math.PI * Math.Sqrt(tmp));
            double daa = a.Value * da / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar AtanPi(in DDScalar a)
        {
            double f = Math.Atan(a.Value) / Math.PI;
            double da = 1.0 / (Math.PI * (a.Value * a.Value + 1.0));
            double daa = -2.0 * a.Value * Math.PI * da * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Exp2(in DDScalar a)
        {
            double f = double.Exp2(a.Value);
            double ln2 = Math.Log(2.0);
            double da = f * ln2;
            double daa = da * ln2;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Exp10(in DDScalar a)
        {
            double f = double.Exp10(a.Value);
            double ln10 = Math.Log(10.0);
            double da = f * ln10;
            double daa = da * ln10;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar ExpM1(in DDScalar a)
        {
            double f = double.ExpM1(a.Value);
            double da = f + 1.0;
            double daa = da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar LogP1(in DDScalar a)
        {
            double f = double.LogP1(a.Value);
            double da = 1.0 / (a.Value + 1.0);
            double daa = -da * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Asinh(in DDScalar a)
        {
            double f = Math.Asinh(a.Value);
            double tmp = a.Value * a.Value + 1.0;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = -a.Value * da / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Acosh(in DDScalar a)
        {
            double f = Math.Acosh(a.Value);
            double tmp = a.Value * a.Value - 1.0;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = -a.Value * da / tmp;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Atanh(in DDScalar a)
        {
            double f = Math.Atanh(a.Value);
            double da = 1.0 / (1.0 - a.Value * a.Value);
            double daa = 2.0 * a.Value * da * da;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar RootN(in DDScalar a, int n)
        {
            if (n == 0) throw new ArgumentException("n cannot be zero", nameof(n));

            double f = Math.Pow(a.Value, 1.0 / n);
            double da = f / (n * a.Value);
            double daa = (1.0 - n) * da / (n * a.Value);

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        /// <summary>
        /// Raises <paramref name="a"/> to a power that is itself an active variable. Unlike the
        /// constant-exponent overload this evaluates <c>log(a)</c>, so it requires a positive base.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Pow(in DDScalar a, in DDScalar b)
        {
            if (a.Size != b.Size || a.Order != b.Order)
                throw new InvalidOperationException("Incompatible sizes or orders for Pow.");

            double f = Math.Pow(a.Value, b.Value);
            double logA = Math.Log(a.Value);
            double da = b.Value * Math.Pow(a.Value, b.Value - 1.0);
            double db = f * logA;
            double daa = b.Value * (b.Value - 1.0) * Math.Pow(a.Value, b.Value - 2.0);
            double dab = Math.Pow(a.Value, b.Value - 1.0) * (1.0 + b.Value * logA);
            double dbb = db * logA;

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Atan2Pi(in DDScalar y, in DDScalar x) => Atan2(y, x) / Math.PI;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Log(in DDScalar a, in DDScalar newBase) => Log(a) / Log(newBase);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (DDScalar Sin, DDScalar Cos) SinCos(in DDScalar a) => (Sin(a), Cos(a));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static (DDScalar SinPi, DDScalar CosPi) SinCosPi(in DDScalar a) => (SinPi(a), CosPi(a));

        /// <summary>
        /// Computes <c>(x * y) + z</c> with a single rounding of the value, the way
        /// <c>IFloatingPointIeee754.FusedMultiplyAdd</c> specifies. The form is bilinear, so its
        /// derivatives are exact regardless: d/dx = y, d/dy = x, d/dz = 1, and the only non-zero
        /// second derivative is the mixed d2/dxdy = 1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar FusedMultiplyAdd(in DDScalar x, in DDScalar y, in DDScalar z)
        {
            if (x.Size != y.Size || x.Order != y.Order || x.Size != z.Size || x.Order != z.Order)
                throw new InvalidOperationException("Incompatible sizes or orders for FusedMultiplyAdd.");

            double f = Math.FusedMultiplyAdd(x.Value, y.Value, z.Value);

            DDScalar result = new DDScalar(x.Size, x.Order);
            Kernel.Ternary<FalseTag,
                ValueCoeff, ValueCoeff, OneCoeff,
                ZeroCoeff, OneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                x.AsReadOnlySpan(), y.AsReadOnlySpan(), z.AsReadOnlySpan(), f,
                new ValueCoeff(y.Value), new ValueCoeff(x.Value), default,
                default, default, default, default, default, default,
                result.AsSpan(), x.Size, x.Order);
            return result;
        }

        /// <summary>
        /// The IEEE 754 remainder <c>a - b * q</c>, where <c>q</c> is <c>a / b</c> rounded to the
        /// nearest integer with ties to even. Same function as <see cref="Math.IEEERemainder"/>,
        /// under the name generic math uses. Piecewise linear, so both second derivatives vanish.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Ieee754Remainder(in DDScalar a, in DDScalar b)
        {
            if (a.Size != b.Size || a.Order != b.Order)
                throw new InvalidOperationException("Incompatible sizes or orders for Ieee754Remainder.");

            double quotient = Math.Round(a.Value / b.Value, MidpointRounding.ToEven);
            double f = Math.IEEERemainder(a.Value, b.Value);

            DDScalar result = new DDScalar(a.Size, a.Order);
            Kernel.Binary<FalseTag, OneCoeff, ValueCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                default, new ValueCoeff(-quotient), default, default, default,
                result.AsSpan(), a.Size, a.Order);
            return result;
        }

        #endregion
    }

    /// <summary>
    /// Represents a generic 3D vector supporting .NET Generic Math.
    /// Can be used with standard types (double, float) or dual numbers (DDScalar3&lt;double&gt; etc.).
    /// </summary>
    public struct Vector3D<T> where T : IFloatingPoint<T>, IRootFunctions<T>
    {
        public T X;
        public T Y;
        public T Z;

        public Vector3D(T x, T y, T z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator +(in Vector3D<T> a, in Vector3D<T> b)
        {
            return new Vector3D<T>(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator -(in Vector3D<T> a, in Vector3D<T> b)
        {
            return new Vector3D<T>(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator -(in Vector3D<T> a)
        {
            return new Vector3D<T>(-a.X, -a.Y, -a.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator *(in Vector3D<T> a, T scalar)
        {
            return new Vector3D<T>(a.X * scalar, a.Y * scalar, a.Z * scalar);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator *(T scalar, in Vector3D<T> a)
        {
            return new Vector3D<T>(scalar * a.X, scalar * a.Y, scalar * a.Z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> operator /(in Vector3D<T> a, T scalar)
        {
            return new Vector3D<T>(a.X / scalar, a.Y / scalar, a.Z / scalar);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly T Dot(in Vector3D<T> b)
        {
            return X * b.X + Y * b.Y + Z * b.Z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Dot(in Vector3D<T> a, in Vector3D<T> b)
        {
            return a.Dot(b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly Vector3D<T> Cross(in Vector3D<T> b)
        {
            return new Vector3D<T>(
                Y * b.Z - Z * b.Y,
                Z * b.X - X * b.Z,
                X * b.Y - Y * b.X
            );
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3D<T> Cross(in Vector3D<T> a, in Vector3D<T> b)
        {
            return a.Cross(b);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly T LengthSquared()
        {
            return X * X + Y * Y + Z * Z;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly T Length()
        {
            return T.Sqrt(LengthSquared());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly Vector3D<T> Normalize()
        {
            T len = Length();
            if (len == T.Zero) return this;
            return this / len;
        }

        public override readonly string ToString()
        {
            return $"({X}, {Y}, {Z})";
        }
    }
}
