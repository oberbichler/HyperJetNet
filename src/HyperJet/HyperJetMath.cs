using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace HyperJet
{
    /// <summary>
    /// Provides highly optimized mathematical operations for both static (<see cref="DDScalar2"/>) and dynamic (<see cref="DDScalar"/>) dual numbers.
    /// Mirroring the standard <see cref="System.Math"/> class.
    /// </summary>
    public static class HyperJetMath
    {
        #region DDScalar2 Functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Sin(in DDScalar2 a)
        {
            double f = Math.Sin(a.Value);
            double da = Math.Cos(a.Value);
            double daa = -f;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Cos(in DDScalar2 a)
        {
            double f = Math.Cos(a.Value);
            double da = -Math.Sin(a.Value);
            double daa = -f;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Tan(in DDScalar2 a)
        {
            double f = Math.Tan(a.Value);
            double da = f * f + 1.0;
            double daa = da * 2.0 * f;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Asin(in DDScalar2 a)
        {
            double f = Math.Asin(a.Value);
            double tmp = 1.0 - a.Value * a.Value;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = da * a.Value / tmp;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Acos(in DDScalar2 a)
        {
            double f = Math.Acos(a.Value);
            double tmp = 1.0 - a.Value * a.Value;
            double da = -1.0 / Math.Sqrt(tmp);
            double daa = da * a.Value / tmp;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Atan(in DDScalar2 a)
        {
            double f = Math.Atan(a.Value);
            double da = 1.0 / (a.Value * a.Value + 1.0);
            double daa = -da * da * 2.0 * a.Value;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Atan2(in DDScalar2 y, in DDScalar2 x)
        {
            double tmp = y.Value * y.Value + x.Value * x.Value;
            double f = Math.Atan2(y.Value, x.Value);
            double da = x.Value / tmp;
            double db = -y.Value / tmp;
            double daa = db * da * 2.0;
            double dab = db * db - da * da;
            double dbb = -daa;

            DDScalar2 result = default;
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                y.AsReadOnlySpan(), x.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Exp(in DDScalar2 a)
        {
            double f = Math.Exp(a.Value);
            double da = f;
            double daa = f;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Log(in DDScalar2 a)
        {
            double f = Math.Log(a.Value);
            double da = 1.0 / a.Value;
            double daa = -da * da;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Log10(in DDScalar2 a)
        {
            double f = Math.Log10(a.Value);
            double ln10 = Math.Log(10.0);
            double da = 1.0 / (a.Value * ln10);
            double daa = -da / a.Value;

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Sqrt(in DDScalar2 a)
        {
            double f = Math.Sqrt(a.Value);
            double da = 1.0 / (2.0 * f);
            double daa = -da / (2.0 * a.Value);

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Pow(in DDScalar2 a, double b)
        {
            double f = Math.Pow(a.Value, b);
            double da = b * Math.Pow(a.Value, b - 1.0);
            double daa = (b - 1.0) * b * Math.Pow(a.Value, b - 2.0);

            DDScalar2 result = default;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Hypot(in DDScalar2 a, in DDScalar2 b)
        {
            double f = Math.Sqrt(a.Value * a.Value + b.Value * b.Value);
            double f3 = f * f * f;
            double da = a.Value / f;
            double db = b.Value / f;
            double daa = b.Value * b.Value / f3;
            double dab = -a.Value * b.Value / f3;
            double dbb = a.Value * a.Value / f3;

            DDScalar2 result = default;
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), DDScalar2.Size, DDScalar2.Order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar2 Abs(in DDScalar2 a)
        {
            return a.Value < 0 ? -a : a;
        }

        #endregion

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
