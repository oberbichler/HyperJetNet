using System;
using System.Runtime.CompilerServices;

namespace HyperJet
{
    /// <summary>
    /// Represents a transient, stack-allocated dual number working on stackalloc or span buffers (100% Zero-Allocation).
    /// </summary>
    public ref struct DDScalarSpan
    {
        private readonly Span<double> _data;
        private readonly int _size;
        private readonly int _order;

        public readonly int Size => _size;
        public readonly int Order => _order;
        public readonly int DataLength => _data.Length;

        public double Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data.Length > 0 ? _data[0] : 0.0;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if (_data.Length == 0) throw new InvalidOperationException("Uninitialized DDScalarSpan");
                _data[0] = value;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double G(int i)
        {
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            return _data[1 + i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetG(int i, double value)
        {
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            _data[1 + i] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double H(int i, int j)
        {
            if (_order < 2) throw new InvalidOperationException("Hessian is only available for 2nd order dual numbers.");
            if (i < 0 || i >= _size) throw new ArgumentOutOfRangeException(nameof(i));
            if (j < 0 || j >= _size) throw new ArgumentOutOfRangeException(nameof(j));

            int index = GetHessianIndex(i, j);
            return _data[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void SetH(int i, int j, double value)
        {
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
        public Span<double> AsSpan() => _data;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly ReadOnlySpan<double> AsReadOnlySpan() => _data;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double[] GetGradient()
        {
            double[] g = new double[_size];
            for (int i = 0; i < _size; i++) g[i] = G(i);
            return g;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void GetGradient(Span<double> destination)
        {
            if (destination.Length < _size) throw new ArgumentException("Destination span is too small.", nameof(destination));
            for (int i = 0; i < _size; i++) destination[i] = G(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly double[,] GetHessian()
        {
            if (_order < 2) throw new InvalidOperationException("Hessian is only available for 2nd order dual numbers.");
            double[,] h = new double[_size, _size];
            for (int i = 0; i < _size; i++)
                for (int j = 0; j < _size; j++)
                    h[i, j] = H(i, j);
            return h;
        }

        #region Constructors and Factory Methods

        public DDScalarSpan(Span<double> data, int size, int order = 2)
        {
            int expectedLength = Kernel.GetDataLength(size, order);
            if (data.Length < expectedLength)
                throw new ArgumentException($"Buffer size too small. Expected at least {expectedLength} elements.", nameof(data));

            _data = data.Slice(0, expectedLength);
            _size = size;
            _order = order;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalarSpan Constant(Span<double> data, double value, int size, int order = 2)
        {
            var result = new DDScalarSpan(data, size, order);
            result.Value = value;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static DDScalarSpan Variable(Span<double> data, int index, double value, int size, int order = 2)
        {
            var result = new DDScalarSpan(data, size, order);
            result.Value = value;
            result.SetG(index, 1.0);
            return result;
        }

        #endregion

        #region Compatibility Checks

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckCompatibility(in DDScalarSpan a, in DDScalarSpan b)
        {
            if (a._size != b._size || a._order != b._order)
            {
                throw new InvalidOperationException($"Incompatible DDScalarSpans. A: (size={a._size}, order={a._order}), B: (size={b._size}, order={b._order})");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckCompatibility(in DDScalarSpan a, in DDScalar b)
        {
            if (a._size != b.Size || a._order != b.Order)
            {
                throw new InvalidOperationException($"Incompatible DDScalars. A: (size={a._size}, order={a._order}), B: (size={b.Size}, order={b.Order})");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckDestination(in DDScalarSpan a, in DDScalarSpan dest)
        {
            if (a._size != dest._size || a._order != dest._order)
            {
                throw new InvalidOperationException("Incompatible destination DDScalarSpan.");
            }
        }

        #endregion

        #region Zero-Allocation Math Methods

        public readonly void Negate(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            for (int i = 0; i < DataLength; i++)
            {
                destination._data[i] = -_data[i];
            }
        }

        public readonly void Add(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, destination);
            Kernel.Binary<FalseTag, OneCoeff, OneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                AsReadOnlySpan(), other.AsReadOnlySpan(), Value + other.Value,
                default, default, default, default, default,
                destination.AsSpan(), _size, _order);
        }

        public readonly void Add(double other, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            _data.CopyTo(destination._data);
            destination.Value += other;
        }

        public readonly void Subtract(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, destination);
            Kernel.Binary<FalseTag, OneCoeff, MinusOneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                AsReadOnlySpan(), other.AsReadOnlySpan(), Value - other.Value,
                default, default, default, default, default,
                destination.AsSpan(), _size, _order);
        }

        public readonly void Subtract(double other, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            _data.CopyTo(destination._data);
            destination.Value -= other;
        }

        public readonly void SubtractFrom(double other, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            Kernel.Unary<FalseTag, MinusOneCoeff, ZeroCoeff>(
                AsReadOnlySpan(), other - Value, default, default,
                destination.AsSpan(), _size, _order);
        }

        public readonly void Multiply(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, destination);
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ZeroCoeff, OneCoeff, ZeroCoeff>(
                AsReadOnlySpan(), other.AsReadOnlySpan(), Value * other.Value,
                new ValueCoeff(other.Value), new ValueCoeff(Value), default, default, default,
                destination.AsSpan(), _size, _order);
        }

        public readonly void Multiply(double other, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            Kernel.Unary<FalseTag, ValueCoeff, ZeroCoeff>(
                AsReadOnlySpan(), Value * other, new ValueCoeff(other), default,
                destination.AsSpan(), _size, _order);
        }

        public readonly void Divide(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, destination);
            double tmp = 1.0 / other.Value;
            double f = Value * tmp;
            double da = tmp;
            double db = -Value * tmp * tmp;
            double dab = -tmp * tmp;
            double dbb = 2.0 * Value * tmp * tmp * tmp;

            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ZeroCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), other.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), default, new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Divide(double other, DDScalarSpan destination)
        {
            Multiply(1.0 / other, destination);
        }

        public readonly void DivideInto(double other, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double tmp = 1.0 / Value;
            double f = other * tmp;
            double db = -other * tmp * tmp;
            double dbb = 2.0 * other * tmp * tmp * tmp;

            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(db), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        #endregion

        #region Zero-Allocation Transcendent Methods

        public readonly void Sin(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Sin(Value);
            double da = Math.Cos(Value);
            double daa = -f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Cos(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Cos(Value);
            double da = -Math.Sin(Value);
            double daa = -f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Tan(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Tan(Value);
            double da = f * f + 1.0;
            double daa = da * 2.0 * f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Asin(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Asin(Value);
            double tmp = 1.0 - Value * Value;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = da * Value / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Acos(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Acos(Value);
            double tmp = 1.0 - Value * Value;
            double da = -1.0 / Math.Sqrt(tmp);
            double daa = da * Value / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Atan(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Atan(Value);
            double da = 1.0 / (Value * Value + 1.0);
            double daa = -da * da * 2.0 * Value;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Atan2(in DDScalarSpan x, DDScalarSpan destination)
        {
            CheckCompatibility(this, x);
            CheckDestination(this, destination);
            double tmp = Value * Value + x.Value * x.Value;
            double f = Math.Atan2(Value, x.Value);
            double da = x.Value / tmp;
            double db = -Value / tmp;
            double daa = db * da * 2.0;
            double dab = db * db - da * da;
            double dbb = -daa;
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), x.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Exp(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Exp(Value);
            double da = f;
            double daa = f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Log(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Log(Value);
            double da = 1.0 / Value;
            double daa = -da * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Log10(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Log10(Value);
            double ln10 = Math.Log(10.0);
            double da = 1.0 / (Value * ln10);
            double daa = -da / Value;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Log2(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Log2(Value);
            double ln2 = Math.Log(2.0);
            double da = 1.0 / (Value * ln2);
            double daa = -da / Value;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Sqrt(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Sqrt(Value);
            double da = 1.0 / (2.0 * f);
            double daa = -da / (2.0 * Value);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Cbrt(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Cbrt(Value);
            double da = 1.0 / (3.0 * f * f);
            double daa = -2.0 * da / (3.0 * Value);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Pow(double b, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Pow(Value, b);
            double da = b * Math.Pow(Value, b - 1.0);
            double daa = (b - 1.0) * b * Math.Pow(Value, b - 2.0);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Hypot(in DDScalarSpan b, DDScalarSpan destination)
        {
            CheckCompatibility(this, b);
            CheckDestination(this, destination);
            double f = Math.Sqrt(Value * Value + b.Value * b.Value);
            double f3 = f * f * f;
            double da = Value / f;
            double db = b.Value / f;
            double daa = b.Value * b.Value / f3;
            double dab = -Value * b.Value / f3;
            double dbb = Value * Value / f3;
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Abs(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            if (Value < 0)
            {
                Negate(destination);
            }
            else
            {
                _data.CopyTo(destination._data);
            }
        }

        public readonly void Sinh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Sinh(Value);
            double da = Math.Cosh(Value);
            double daa = f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Cosh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Cosh(Value);
            double da = Math.Sinh(Value);
            double daa = f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Tanh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Tanh(Value);
            double da = 1.0 - f * f;
            double daa = -2.0 * f * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        #endregion

        #region Standard Operators (Dynamic DDScalar Fallback)

        public static DDScalar operator -(in DDScalarSpan a)
        {
            var result = new DDScalar(a._size, a._order);
            for (int i = 0; i < a.DataLength; i++)
            {
                result.AsSpan()[i] = -a._data[i];
            }
            return result;
        }

        public static DDScalar operator +(in DDScalarSpan a, in DDScalarSpan b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Add(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator +(in DDScalarSpan a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Add(new DDScalarSpan(b.AsSpan(), b.Size, b.Order), new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator +(in DDScalar a, in DDScalarSpan b) => b + a;

        public static DDScalar operator +(in DDScalarSpan a, double b)
        {
            var result = new DDScalar(a._size, a._order);
            a.Add(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator +(double a, in DDScalarSpan b) => b + a;

        public static DDScalar operator -(in DDScalarSpan a, in DDScalarSpan b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Subtract(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator -(in DDScalarSpan a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Subtract(new DDScalarSpan(b.AsSpan(), b.Size, b.Order), new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator -(in DDScalar a, in DDScalarSpan b)
        {
            CheckCompatibility(b, a);
            var result = new DDScalar(a.Size, a.Order);
            new DDScalarSpan(a.AsSpan(), a.Size, a.Order).Subtract(b, new DDScalarSpan(result.AsSpan(), a.Size, a.Order));
            return result;
        }

        public static DDScalar operator -(in DDScalarSpan a, double b)
        {
            var result = new DDScalar(a._size, a._order);
            a.Subtract(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator -(double a, in DDScalarSpan b)
        {
            var result = new DDScalar(b._size, b._order);
            b.SubtractFrom(a, new DDScalarSpan(result.AsSpan(), b._size, b._order));
            return result;
        }

        public static DDScalar operator *(in DDScalarSpan a, in DDScalarSpan b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Multiply(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator *(in DDScalarSpan a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Multiply(new DDScalarSpan(b.AsSpan(), b.Size, b.Order), new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator *(in DDScalar a, in DDScalarSpan b) => b * a;

        public static DDScalar operator *(in DDScalarSpan a, double b)
        {
            var result = new DDScalar(a._size, a._order);
            a.Multiply(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator *(double a, in DDScalarSpan b) => b * a;

        public static DDScalar operator /(in DDScalarSpan a, in DDScalarSpan b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Divide(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator /(in DDScalarSpan a, in DDScalar b)
        {
            CheckCompatibility(a, b);
            var result = new DDScalar(a._size, a._order);
            a.Divide(new DDScalarSpan(b.AsSpan(), b.Size, b.Order), new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator /(in DDScalar a, in DDScalarSpan b)
        {
            CheckCompatibility(b, a);
            var result = new DDScalar(a.Size, a.Order);
            new DDScalarSpan(a.AsSpan(), a.Size, a.Order).Divide(b, new DDScalarSpan(result.AsSpan(), a.Size, a.Order));
            return result;
        }

        public static DDScalar operator /(in DDScalarSpan a, double b)
        {
            var result = new DDScalar(a._size, a._order);
            a.Divide(b, new DDScalarSpan(result.AsSpan(), a._size, a._order));
            return result;
        }

        public static DDScalar operator /(double a, in DDScalarSpan b)
        {
            var result = new DDScalar(b._size, b._order);
            b.DivideInto(a, new DDScalarSpan(result.AsSpan(), b._size, b._order));
            return result;
        }

        #endregion

        #region Comparisons

        public static bool operator ==(in DDScalarSpan a, in DDScalarSpan b) => a.Value == b.Value;
        public static bool operator !=(in DDScalarSpan a, in DDScalarSpan b) => a.Value != b.Value;
        public static bool operator <(in DDScalarSpan a, in DDScalarSpan b) => a.Value < b.Value;
        public static bool operator >(in DDScalarSpan a, in DDScalarSpan b) => a.Value > b.Value;
        public static bool operator <=(in DDScalarSpan a, in DDScalarSpan b) => a.Value <= b.Value;
        public static bool operator >=(in DDScalarSpan a, in DDScalarSpan b) => a.Value >= b.Value;

        public static bool operator ==(in DDScalarSpan a, double b) => a.Value == b;
        public static bool operator !=(in DDScalarSpan a, double b) => a.Value != b;
        public static bool operator <(in DDScalarSpan a, double b) => a.Value < b;
        public static bool operator >(in DDScalarSpan a, double b) => a.Value > b;
        public static bool operator <=(in DDScalarSpan a, double b) => a.Value <= b;
        public static bool operator >=(in DDScalarSpan a, double b) => a.Value >= b;

        public static bool operator ==(double a, in DDScalarSpan b) => a == b.Value;
        public static bool operator !=(double a, in DDScalarSpan b) => a != b.Value;
        public static bool operator <(double a, in DDScalarSpan b) => a < b.Value;
        public static bool operator >(double a, in DDScalarSpan b) => a > b.Value;
        public static bool operator <=(double a, in DDScalarSpan b) => a <= b.Value;
        public static bool operator >=(double a, in DDScalarSpan b) => a >= b.Value;

        public override readonly bool Equals(object? obj) => throw new NotSupportedException("Equals on ref struct is not supported.");
        public override readonly int GetHashCode() => throw new NotSupportedException("GetHashCode on ref struct is not supported.");

        #endregion

        public override readonly string ToString()
        {
            if (_data.Length == 0) return "Uninitialized DDScalarSpan";
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
}