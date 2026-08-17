using System;
using System.Runtime.CompilerServices;

namespace HyperJet
{
    /// <summary>
    /// Represents a dynamic, heap-allocated dual number supporting runtime variable size and order (1st or 2nd order).
    /// </summary>
    public struct DDScalar
    {
        private readonly double[] _data;
        private readonly int _size;
        private readonly int _order;

        public readonly int Size => _size;
        public readonly int Order => _order;
        public readonly int DataLength => _data?.Length ?? 0;

        public double Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data != null ? _data[0] : 0.0;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if (_data == null) ThrowNull();
                _data[0] = value;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double G(int i)
        {
            if (_data == null) ThrowNull();
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            return _data[1 + i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetG(int i, double value)
        {
            if (_data == null) ThrowNull();
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            _data[1 + i] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double H(int i, int j)
        {
            if (_data == null) ThrowNull();
            if (_order < 2) throw new InvalidOperationException("Hessian is only available for 2nd order dual numbers.");
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            if (j < 0 || j >= _size) throw new ArgumentOutOfRangeException(nameof(j));

            int index = GetHessianIndex(i, j);
            return _data[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetH(int i, int j, double value)
        {
            if (_data == null) ThrowNull();
            if (_order < 2) throw new InvalidOperationException("Hessian is only available for 2nd order dual numbers.");
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            if (j < 0 || j >= _size) throw new ArgumentOutOfRangeException(nameof(j));

            int index = GetHessianIndex(i, j);
            _data[index] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private readonly int GetHessianIndex(int i, int j)
        {
            if (i < j)
            {
                return 1 + _size + (2 * _size - 1 - i) * i / 2 + j;
            }
            else
            {
                return 1 + _size + (2 * _size - 1 - j) * j / 2 + i;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<double> AsSpan() => _data != null ? _data : Span<double>.Empty;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly ReadOnlySpan<double> AsReadOnlySpan() => _data != null ? _data : ReadOnlySpan<double>.Empty;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double[] GetGradient()
        {
            if (_data == null) ThrowNull();
            double[] g = new double[_size];
            for (int i = 0; i < _size; i++) g[i] = G(i);
            return g;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void GetGradient(Span<double> destination)
        {
            if (_data == null) ThrowNull();
            if (destination.Length < _size) throw new ArgumentException("Destination span is too small.", nameof(destination));
            for (int i = 0; i < _size; i++) destination[i] = G(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double[,] GetHessian()
        {
            if (_data == null) ThrowNull();
            if (_order < 2) throw new InvalidOperationException("Hessian is only available for 2nd order dual numbers.");
            double[,] h = new double[_size, _size];
            for (int i = 0; i < _size; i++)
                for (int j = 0; j < _size; j++)
                    h[i, j] = H(i, j);
            return h;
        }

        /// <summary>
        /// Evaluates the Taylor expansion of the represented function around the point it was
        /// evaluated at: <c>f(x + d) = f(x) + grad(f) . d + 1/2 d^T H d</c>.
        /// </summary>
        /// <param name="d">The offset from the expansion point, one component per variable.</param>
        /// <remarks>
        /// For a 1st-order scalar the quadratic term is absent and this is the linear model. For a
        /// function that is itself quadratic the 2nd-order expansion is exact; otherwise the error
        /// is O(|d|^3). This is the local model used by a trust-region step or a line search.
        /// </remarks>
        public readonly double Evaluate(params ReadOnlySpan<double> d)
        {
            if (_data == null) ThrowNull();
            if (d.Length != _size)
                throw new ArgumentException($"Expected {_size} offsets, got {d.Length}.", nameof(d));

            return EvaluateTaylor(_data, d, _size, _order);
        }

        /// <summary>
        /// Shared Taylor evaluation over the packed coefficient layout: value, gradient, then the
        /// upper triangle of the Hessian in row order.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static double EvaluateTaylor(ReadOnlySpan<double> data, ReadOnlySpan<double> d, int size, int order)
        {
            double result = data[0];

            for (int i = 0; i < size; i++)
            {
                result += data[1 + i] * d[i];
            }

            if (order < 2) return result;

            // One pass over the stored triangle reaches H[i,i] first and then H[i,j] for j > i. Each
            // off-diagonal coefficient stands for two symmetric terms of the quadratic form, which
            // is exactly what cancels the one half.
            int k = 1 + size;
            for (int i = 0; i < size; i++)
            {
                double di = d[i];
                result += 0.5 * data[k++] * di * di;

                for (int j = i + 1; j < size; j++)
                {
                    result += data[k++] * di * d[j];
                }
            }

            return result;
        }

        #region Constructors and Factory Methods

        public DDScalar(int size, int order = 2)
        {
            if (size < 0) throw new ArgumentOutOfRangeException(nameof(size));
            if (order < 1 || order > 2) throw new ArgumentOutOfRangeException(nameof(order));

            _size = size;
            _order = order;
            _data = new double[Kernel.GetDataLength(size, order)];
        }

        private DDScalar(double[] data, int size, int order)
        {
            _data = data;
            _size = size;
            _order = order;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Constant(double value, int size, int order = 2)
        {
            var result = new DDScalar(size, order);
            result.Value = value;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar Variable(int index, double value, int size, int order = 2)
        {
            var result = new DDScalar(size, order);
            result.Value = value;
            result.SetG(index, 1.0);
            return result;
        }

        public static DDScalar[] Variables(double[] values, int order = 2)
        {
            int size = values.Length;
            DDScalar[] result = new DDScalar[size];
            for (int i = 0; i < size; i++)
            {
                result[i] = Variable(i, values[i], size, order);
            }
            return result;
        }

        #endregion

        #region Core Checks

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckCompatibility(in DDScalar a, in DDScalar b)
        {
            if (a._size != b._size || a._order != b._order)
            {
                throw new InvalidOperationException($"Incompatible DDScalars. A: (size={a._size}, order={a._order}), B: (size={b._size}, order={b._order})");
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [System.Diagnostics.CodeAnalysis.DoesNotReturn]
        private static void ThrowNull()
        {
            throw new InvalidOperationException("The DDScalar has not been properly initialized.");
        }

        #endregion

        #region Operators

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator -(in DDScalar a)
        {
            if (a._data == null) ThrowNull();
            var result = new DDScalar(a._size, a._order);
            for (int i = 0; i < a.DataLength; i++)
            {
                result._data[i] = -a._data[i];
            }
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator +(in DDScalar a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            Kernel.Binary<FalseTag, OneCoeff, OneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), a.Value + b.Value,
                default, default, default, default, default,
                result.AsSpan(), a._size, a._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator +(in DDScalar a, double b)
        {
            if (a._data == null) ThrowNull();
            var result = new DDScalar(a._size, a._order);
            a._data.CopyTo(result._data, 0);
            result.Value += b;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator +(double a, in DDScalar b) => b + a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator -(in DDScalar a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            Kernel.Binary<FalseTag, OneCoeff, MinusOneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), a.Value - b.Value,
                default, default, default, default, default,
                result.AsSpan(), a._size, a._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator -(in DDScalar a, double b)
        {
            if (a._data == null) ThrowNull();
            var result = new DDScalar(a._size, a._order);
            a._data.CopyTo(result._data, 0);
            result.Value -= b;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator -(double a, in DDScalar b)
        {
            if (b._data == null) ThrowNull();
            var result = new DDScalar(b._size, b._order);
            Kernel.Unary<FalseTag, MinusOneCoeff, ZeroCoeff>(
                b.AsReadOnlySpan(), a - b.Value, default, default,
                result.AsSpan(), b._size, b._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator *(in DDScalar a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ZeroCoeff, OneCoeff, ZeroCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), a.Value * b.Value,
                new ValueCoeff(b.Value), new ValueCoeff(a.Value), default, default, default,
                result.AsSpan(), a._size, a._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator *(in DDScalar a, double b)
        {
            if (a._data == null) ThrowNull();
            var result = new DDScalar(a._size, a._order);
            Kernel.Unary<FalseTag, ValueCoeff, ZeroCoeff>(
                a.AsReadOnlySpan(), a.Value * b, new ValueCoeff(b), default,
                result.AsSpan(), a._size, a._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator *(double a, in DDScalar b) => b * a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator /(in DDScalar a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            double tmp = 1.0 / b.Value;
            double f = a.Value * tmp;
            double da = tmp;
            double db = -a.Value * tmp * tmp;
            double dab = -tmp * tmp;
            double dbb = 2.0 * a.Value * tmp * tmp * tmp;

            var result = new DDScalar(a._size, a._order);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ZeroCoeff, ValueCoeff, ValueCoeff>(
                a.AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), default, new ValueCoeff(dab), new ValueCoeff(dbb),
                result.AsSpan(), a._size, a._order);
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator /(in DDScalar a, double b) => a * (1.0 / b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalar operator /(double a, in DDScalar b)
        {
            if (b._data == null) ThrowNull();
            double tmp = 1.0 / b.Value;
            double f = a * tmp;
            double db = -a * tmp * tmp;
            double dbb = 2.0 * a * tmp * tmp * tmp;

            var result = new DDScalar(b._size, b._order);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                b.AsReadOnlySpan(), f, new ValueCoeff(db), new ValueCoeff(dbb),
                result.AsSpan(), b._size, b._order);
            return result;
        }

        #endregion

        #region Comparisons

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in DDScalar a, in DDScalar b) => a.Value == b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in DDScalar a, in DDScalar b) => a.Value != b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <(in DDScalar a, in DDScalar b) => a.Value < b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >(in DDScalar a, in DDScalar b) => a.Value > b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <=(in DDScalar a, in DDScalar b) => a.Value <= b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >=(in DDScalar a, in DDScalar b) => a.Value >= b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(in DDScalar a, double b) => a.Value == b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(in DDScalar a, double b) => a.Value != b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <(in DDScalar a, double b) => a.Value < b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >(in DDScalar a, double b) => a.Value > b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <=(in DDScalar a, double b) => a.Value <= b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >=(in DDScalar a, double b) => a.Value >= b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(double a, in DDScalar b) => a == b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(double a, in DDScalar b) => a != b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <(double a, in DDScalar b) => a < b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >(double a, in DDScalar b) => a > b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator <=(double a, in DDScalar b) => a <= b.Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator >=(double a, in DDScalar b) => a >= b.Value;

        public override readonly bool Equals(object? obj) => obj is DDScalar other && this == other;

        public override readonly int GetHashCode() => Value.GetHashCode();

        #endregion

        public override readonly string ToString()
        {
            if (_data == null) return "Uninitialized DDScalar";
            string res = $"{Value}";
            if (_size > 0)
            {
                res += " [g: (";
                for (int i = 0; i < _size; i++)
                {
                    res += G(i).ToString() + (i < _size - 1 ? ", " : "");
                }
                res += ")";
                if (_order >= 2)
                {
                    res += ", H: (";
                    for (int i = 0; i < _size; i++)
                    {
                        res += "(";
                        for (int j = 0; j < _size; j++)
                        {
                            res += H(i, j).ToString() + (j < _size - 1 ? ", " : "");
                        }
                        res += ")" + (i < _size - 1 ? ", " : "");
                    }
                    res += ")";
                }
                res += "]";
            }
            return res;
        }
    }

    /// <summary>
    /// Provides extension methods for tuple deconstruction on arrays and spans.
    /// </summary>
    public static class HyperJetExtensions
    {
        #region Array Deconstructors

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2)
        {
            v1 = array[0]; v2 = array[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9]; v11 = array[10];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9]; v11 = array[10]; v12 = array[11];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9]; v11 = array[10]; v12 = array[11]; v13 = array[12];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9]; v11 = array[10]; v12 = array[11]; v13 = array[12]; v14 = array[13];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this T[] array, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14, out T v15)
        {
            v1 = array[0]; v2 = array[1]; v3 = array[2]; v4 = array[3]; v5 = array[4]; v6 = array[5]; v7 = array[6]; v8 = array[7]; v9 = array[8]; v10 = array[9]; v11 = array[10]; v12 = array[11]; v13 = array[12]; v14 = array[13]; v15 = array[14];
        }

        #endregion

        #region Span Deconstructors

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2)
        {
            v1 = span[0]; v2 = span[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12]; v14 = span[13];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this Span<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14, out T v15)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12]; v14 = span[13]; v15 = span[14];
        }

        #endregion

        #region ReadOnlySpan Deconstructors

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2)
        {
            v1 = span[0]; v2 = span[1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12]; v14 = span[13];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Deconstruct<T>(this ReadOnlySpan<T> span, out T v1, out T v2, out T v3, out T v4, out T v5, out T v6, out T v7, out T v8, out T v9, out T v10, out T v11, out T v12, out T v13, out T v14, out T v15)
        {
            v1 = span[0]; v2 = span[1]; v3 = span[2]; v4 = span[3]; v5 = span[4]; v6 = span[5]; v7 = span[6]; v8 = span[7]; v9 = span[8]; v10 = span[9]; v11 = span[10]; v12 = span[11]; v13 = span[12]; v14 = span[13]; v15 = span[14];
        }

        #endregion
    }
}
