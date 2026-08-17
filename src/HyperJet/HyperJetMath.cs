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

        /// <summary>
        /// Identical to <see cref="Ieee754Remainder(in DDScalar, in DDScalar)"/>. .NET carries this
        /// function under two names — <see cref="Math.IEEERemainder"/> predates the naming guideline
        /// that gave the generic-math surface <c>Ieee754Remainder</c> — and this facade mirrors
        /// <see cref="Math"/>, so it offers the spelling a caller coming from there will reach for.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar IEEERemainder(in DDScalar a, in DDScalar b) => Ieee754Remainder(a, b);

        #endregion

        #region Value-Level Helpers

        // IsCanonical, IsComplexNumber, IsImaginaryNumber and IsZero are explicit INumberBase
        // implementations on double and cannot be called as double.IsX(...); a constrained type
        // parameter reaches them.
        private static bool IsCanonicalHelper<T>(T v) where T : INumberBase<T> => T.IsCanonical(v);
        private static bool IsComplexNumberHelper<T>(T v) where T : INumberBase<T> => T.IsComplexNumber(v);
        private static bool IsImaginaryNumberHelper<T>(T v) where T : INumberBase<T> => T.IsImaginaryNumber(v);
        private static bool IsZeroHelper<T>(T v) where T : INumberBase<T> => T.IsZero(v);

        /// <summary>
        /// A fresh scalar holding the same coefficients. Selection helpers copy rather than hand back
        /// an operand, because <see cref="DDScalar"/> shares its buffer between struct copies and
        /// every other operation in the library returns an independent result.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static DDScalar CopyOf(in DDScalar a)
        {
            DDScalar result = new DDScalar(a.Size, a.Order);
            a.AsReadOnlySpan().CopyTo(result.AsSpan());
            return result;
        }

        /// <summary>A constant carrying <paramref name="value"/> and no derivatives.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static DDScalar ConstantLike(in DDScalar a, double value) => DDScalar.Constant(value, a.Size, a.Order);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckSameShape(in DDScalar a, in DDScalar b, string what)
        {
            if (a.Size != b.Size || a.Order != b.Order)
                throw new InvalidOperationException($"Incompatible sizes or orders for {what}.");
        }

        /// <summary>Selects an operand by value and returns a copy of it, derivatives included.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Min(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "Min");
            return CopyOf(x.Value < y.Value ? x : y);
        }

        /// <summary>Selects an operand by value and returns a copy of it, derivatives included.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Max(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "Max");
            return CopyOf(x.Value > y.Value ? x : y);
        }

        /// <summary>Selects an operand by value and returns a copy of it, derivatives included.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MinMagnitude(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MinMagnitude");
            return CopyOf(Math.Abs(x.Value) < Math.Abs(y.Value) ? x : y);
        }

        /// <summary>Selects an operand by value and returns a copy of it, derivatives included.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MaxMagnitude(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MaxMagnitude");
            return CopyOf(Math.Abs(x.Value) >= Math.Abs(y.Value) ? x : y);
        }

        /// <summary>As <see cref="Min"/>, but a NaN operand loses against a number.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MinNumber(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MinNumber");
            if (double.IsNaN(x.Value)) return CopyOf(y);
            if (double.IsNaN(y.Value)) return CopyOf(x);
            return Min(x, y);
        }

        /// <summary>As <see cref="Max"/>, but a NaN operand loses against a number.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MaxNumber(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MaxNumber");
            if (double.IsNaN(x.Value)) return CopyOf(y);
            if (double.IsNaN(y.Value)) return CopyOf(x);
            return Max(x, y);
        }

        /// <summary>As <see cref="MinMagnitude"/>, but a NaN operand loses against a number.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MinMagnitudeNumber(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MinMagnitudeNumber");
            if (double.IsNaN(x.Value)) return CopyOf(y);
            if (double.IsNaN(y.Value)) return CopyOf(x);
            return MinMagnitude(x, y);
        }

        /// <summary>As <see cref="MaxMagnitude"/>, but a NaN operand loses against a number.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar MaxMagnitudeNumber(in DDScalar x, in DDScalar y)
        {
            CheckSameShape(x, y, "MaxMagnitudeNumber");
            if (double.IsNaN(x.Value)) return CopyOf(y);
            if (double.IsNaN(y.Value)) return CopyOf(x);
            return MaxMagnitude(x, y);
        }

        /// <summary>Selects value, min or max and returns a copy of it, derivatives included.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Clamp(in DDScalar value, in DDScalar min, in DDScalar max)
        {
            CheckSameShape(value, min, "Clamp");
            CheckSameShape(value, max, "Clamp");
            if (min.Value > max.Value) throw new ArgumentException("min cannot be greater than max");

            if (value.Value < min.Value) return CopyOf(min);
            if (value.Value > max.Value) return CopyOf(max);
            return CopyOf(value);
        }

        /// <summary>The sign as -1, 0 or 1. Piecewise constant, so the result carries no derivatives.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Sign(in DDScalar value) => ConstantLike(value, Math.Sign(value.Value));

        /// <summary>The magnitude of <paramref name="value"/> with the sign of <paramref name="sign"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar CopySign(in DDScalar value, in DDScalar sign)
        {
            CheckSameShape(value, sign, "CopySign");
            return double.IsNegative(sign.Value) ? -Abs(value) : Abs(value);
        }

        // Rounding is piecewise constant, so away from the break points the derivative is zero
        // and the result is a constant.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Round(in DDScalar x) => ConstantLike(x, Math.Round(x.Value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Round(in DDScalar x, int digits, MidpointRounding mode) => ConstantLike(x, Math.Round(x.Value, digits, mode));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Round(in DDScalar x, MidpointRounding mode) => ConstantLike(x, Math.Round(x.Value, mode));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Floor(in DDScalar x) => ConstantLike(x, Math.Floor(x.Value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Ceiling(in DDScalar x) => ConstantLike(x, Math.Ceiling(x.Value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Truncate(in DDScalar x) => ConstantLike(x, Math.Truncate(x.Value));

        /// <summary>
        /// The neighbouring representable value. Within a binade the step is a constant, so this is
        /// <c>x + c</c> and the derivatives survive it — unlike the piecewise-constant rounding above.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar BitIncrement(in DDScalar x)
        {
            DDScalar result = CopyOf(x);
            result.AsSpan()[0] = double.BitIncrement(x.Value);
            return result;
        }

        /// <summary>The next representable value below, keeping the derivatives.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar BitDecrement(in DDScalar x)
        {
            DDScalar result = CopyOf(x);
            result.AsSpan()[0] = double.BitDecrement(x.Value);
            return result;
        }

        /// <summary>The base-2 exponent of the value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ILogB(in DDScalar x) => double.ILogB(x.Value);

        /// <summary>
        /// Multiplies by <c>2^n</c>. Linear, so every coefficient scales with it; using ScaleB per
        /// coefficient keeps the scaling exact.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar ScaleB(in DDScalar x, int n)
        {
            DDScalar result = new DDScalar(x.Size, x.Order);

            ReadOnlySpan<double> source = x.AsReadOnlySpan();
            Span<double> destination = result.AsSpan();

            for (int i = 0; i < source.Length; i++) destination[i] = Math.ScaleB(source[i], n);

            return result;
        }

        // Classification looks only at the value; the derivatives cannot change the answer.

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsCanonical(in DDScalar x) => IsCanonicalHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsComplexNumber(in DDScalar x) => IsComplexNumberHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEvenInteger(in DDScalar x) => double.IsEvenInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsFinite(in DDScalar x) => double.IsFinite(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsImaginaryNumber(in DDScalar x) => IsImaginaryNumberHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsInfinity(in DDScalar x) => double.IsInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsInteger(in DDScalar x) => double.IsInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNaN(in DDScalar x) => double.IsNaN(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNegative(in DDScalar x) => double.IsNegative(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNegativeInfinity(in DDScalar x) => double.IsNegativeInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNormal(in DDScalar x) => double.IsNormal(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsOddInteger(in DDScalar x) => double.IsOddInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPositive(in DDScalar x) => double.IsPositive(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPositiveInfinity(in DDScalar x) => double.IsPositiveInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsRealNumber(in DDScalar x) => double.IsRealNumber(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSubnormal(in DDScalar x) => double.IsSubnormal(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsZero(in DDScalar x) => IsZeroHelper(x.Value);

        #endregion

        #region DDScalarSpan Queries

        // Classification and the exponent are plain value queries, so they live here as statics for
        // both dynamic models rather than as instance methods on one of them.

        /// <summary>The base-2 exponent of the value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ILogB(in DDScalarSpan x) => double.ILogB(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsCanonical(in DDScalarSpan x) => IsCanonicalHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsComplexNumber(in DDScalarSpan x) => IsComplexNumberHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEvenInteger(in DDScalarSpan x) => double.IsEvenInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsFinite(in DDScalarSpan x) => double.IsFinite(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsImaginaryNumber(in DDScalarSpan x) => IsImaginaryNumberHelper(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsInfinity(in DDScalarSpan x) => double.IsInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsInteger(in DDScalarSpan x) => double.IsInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNaN(in DDScalarSpan x) => double.IsNaN(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNegative(in DDScalarSpan x) => double.IsNegative(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNegativeInfinity(in DDScalarSpan x) => double.IsNegativeInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNormal(in DDScalarSpan x) => double.IsNormal(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsOddInteger(in DDScalarSpan x) => double.IsOddInteger(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPositive(in DDScalarSpan x) => double.IsPositive(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPositiveInfinity(in DDScalarSpan x) => double.IsPositiveInfinity(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsRealNumber(in DDScalarSpan x) => double.IsRealNumber(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSubnormal(in DDScalarSpan x) => double.IsSubnormal(x.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsZero(in DDScalarSpan x) => IsZeroHelper(x.Value);

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
