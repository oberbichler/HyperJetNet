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
            if (d.Length != _size)
                throw new ArgumentException($"Expected {_size} offsets, got {d.Length}.", nameof(d));

            return DDScalar.EvaluateTaylor(_data, d, _size, _order);
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

            if (dest._data.Overlaps(a._data)) ThrowOverlap();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckDestination(in DDScalarSpan a, in DDScalarSpan b, in DDScalarSpan dest)
        {
            CheckDestination(a, dest);

            if (dest._data.Overlaps(b._data)) ThrowOverlap();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CheckDestination(in DDScalarSpan a, in DDScalarSpan b, in DDScalarSpan c, in DDScalarSpan dest)
        {
            CheckDestination(a, b, dest);

            if (dest._data.Overlaps(c._data)) ThrowOverlap();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        [System.Diagnostics.CodeAnalysis.DoesNotReturn]
        private static void ThrowOverlap()
        {
            throw new ArgumentException(
                "The destination must not overlap an operand. The kernels read the operands while " +
                "writing the destination, so evaluating in place would corrupt the derivatives.",
                "destination");
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
            CheckDestination(this, other, destination);
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
            CheckDestination(this, other, destination);
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
            CheckDestination(this, other, destination);
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
            CheckDestination(this, other, destination);
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
            CheckDestination(this, x, destination);
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
            CheckDestination(this, b, destination);
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

        public readonly void SinPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Sin(Math.PI * Value);
            double da = Math.PI * Math.Cos(Math.PI * Value);
            double daa = -Math.PI * Math.PI * f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void CosPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Cos(Math.PI * Value);
            double da = -Math.PI * Math.Sin(Math.PI * Value);
            double daa = -Math.PI * Math.PI * f;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void TanPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Tan(Math.PI * Value);
            double da = Math.PI * (f * f + 1.0);
            double daa = 2.0 * Math.PI * f * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void AsinPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Asin(Value) / Math.PI;
            double tmp = 1.0 - Value * Value;
            double da = 1.0 / (Math.PI * Math.Sqrt(tmp));
            double daa = Value * da / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void AcosPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Acos(Value) / Math.PI;
            double tmp = 1.0 - Value * Value;
            double da = -1.0 / (Math.PI * Math.Sqrt(tmp));
            double daa = Value * da / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void AtanPi(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Atan(Value) / Math.PI;
            double da = 1.0 / (Math.PI * (Value * Value + 1.0));
            double daa = -2.0 * Value * Math.PI * da * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Exp2(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = double.Exp2(Value);
            double ln2 = Math.Log(2.0);
            double da = f * ln2;
            double daa = da * ln2;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Exp10(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = double.Exp10(Value);
            double ln10 = Math.Log(10.0);
            double da = f * ln10;
            double daa = da * ln10;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void ExpM1(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = double.ExpM1(Value);
            double da = f + 1.0;
            double daa = da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void LogP1(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = double.LogP1(Value);
            double da = 1.0 / (Value + 1.0);
            double daa = -da * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Asinh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Asinh(Value);
            double tmp = Value * Value + 1.0;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = -Value * da / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Acosh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Acosh(Value);
            double tmp = Value * Value - 1.0;
            double da = 1.0 / Math.Sqrt(tmp);
            double daa = -Value * da / tmp;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void Atanh(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            double f = Math.Atanh(Value);
            double da = 1.0 / (1.0 - Value * Value);
            double daa = 2.0 * Value * da * da;
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        public readonly void RootN(int n, DDScalarSpan destination)
        {
            if (n == 0) throw new ArgumentException("n cannot be zero", nameof(n));
            CheckDestination(this, destination);
            double f = Math.Pow(Value, 1.0 / n);
            double da = f / (n * Value);
            double daa = (1.0 - n) * da / (n * Value);
            Kernel.Unary<FalseTag, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), f, new ValueCoeff(da), new ValueCoeff(daa),
                destination.AsSpan(), _size, _order);
        }

        /// <summary>
        /// Raises this value to a power that is itself an active variable. Unlike the
        /// constant-exponent overload this evaluates <c>log(a)</c>, so it requires a positive base.
        /// </summary>
        public readonly void Pow(in DDScalarSpan b, DDScalarSpan destination)
        {
            CheckCompatibility(this, b);
            CheckDestination(this, b, destination);
            double f = Math.Pow(Value, b.Value);
            double logA = Math.Log(Value);
            double da = b.Value * Math.Pow(Value, b.Value - 1.0);
            double db = f * logA;
            double daa = b.Value * (b.Value - 1.0) * Math.Pow(Value, b.Value - 2.0);
            double dab = Math.Pow(Value, b.Value - 1.0) * (1.0 + b.Value * logA);
            double dbb = db * logA;
            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        /// <summary>Writes atan2(this, x) / pi into <paramref name="destination"/>.</summary>
        public readonly void Atan2Pi(in DDScalarSpan x, DDScalarSpan destination)
        {
            CheckCompatibility(this, x);
            CheckDestination(this, x, destination);

            // Every derivative of atan2 is simply scaled by 1/pi.
            double tmp = Value * Value + x.Value * x.Value;
            double f = Math.Atan2(Value, x.Value) / Math.PI;
            double dy = x.Value / tmp;
            double dx = -Value / tmp;
            double da = dy / Math.PI;
            double db = dx / Math.PI;
            double daa = dx * dy * 2.0 / Math.PI;
            double dab = (dx * dx - dy * dy) / Math.PI;
            double dbb = -daa;

            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), x.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        /// <summary>Writes log(this) / log(newBase) into <paramref name="destination"/>.</summary>
        public readonly void Log(in DDScalarSpan newBase, DDScalarSpan destination)
        {
            CheckCompatibility(this, newBase);
            CheckDestination(this, newBase, destination);

            double f = Math.Log(Value) / Math.Log(newBase.Value);
            double la = Math.Log(Value);
            double lb = Math.Log(newBase.Value);
            double da = 1.0 / (Value * lb);
            double db = -la / (newBase.Value * lb * lb);
            double daa = -da / Value;
            double dab = -1.0 / (Value * newBase.Value * lb * lb);
            double dbb = la * (2.0 + lb) / (newBase.Value * newBase.Value * lb * lb * lb);

            Kernel.Binary<FalseTag, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff, ValueCoeff>(
                AsReadOnlySpan(), newBase.AsReadOnlySpan(), f,
                new ValueCoeff(da), new ValueCoeff(db), new ValueCoeff(daa), new ValueCoeff(dab), new ValueCoeff(dbb),
                destination.AsSpan(), _size, _order);
        }

        /// <summary>Writes sin and cos into two distinct destinations.</summary>
        public readonly void SinCos(DDScalarSpan sinDestination, DDScalarSpan cosDestination)
        {
            if (sinDestination._data.Overlaps(cosDestination._data)) ThrowOverlap();
            Sin(sinDestination);
            Cos(cosDestination);
        }

        /// <summary>Writes sin(pi*a) and cos(pi*a) into two distinct destinations.</summary>
        public readonly void SinCosPi(DDScalarSpan sinDestination, DDScalarSpan cosDestination)
        {
            if (sinDestination._data.Overlaps(cosDestination._data)) ThrowOverlap();
            SinPi(sinDestination);
            CosPi(cosDestination);
        }

        /// <summary>
        /// Computes <c>(this * y) + z</c> with a single rounding of the value, the way
        /// <c>IFloatingPointIeee754.FusedMultiplyAdd</c> specifies. The form is bilinear, so its
        /// derivatives are exact regardless: d/dx = y, d/dy = x, d/dz = 1, and the only non-zero
        /// second derivative is the mixed d2/dxdy = 1.
        /// </summary>
        public readonly void FusedMultiplyAdd(in DDScalarSpan y, in DDScalarSpan z, DDScalarSpan destination)
        {
            CheckCompatibility(this, y);
            CheckCompatibility(this, z);
            CheckDestination(this, y, z, destination);

            double f = Math.FusedMultiplyAdd(Value, y.Value, z.Value);

            Kernel.Ternary<FalseTag,
                ValueCoeff, ValueCoeff, OneCoeff,
                ZeroCoeff, OneCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                AsReadOnlySpan(), y.AsReadOnlySpan(), z.AsReadOnlySpan(), f,
                new ValueCoeff(y.Value), new ValueCoeff(Value), default,
                default, default, default, default, default, default,
                destination.AsSpan(), _size, _order);
        }

        /// <summary>
        /// The IEEE 754 remainder <c>this - b * q</c>, where <c>q</c> is <c>this / b</c> rounded to
        /// the nearest integer with ties to even. Same function as <see cref="Math.IEEERemainder"/>,
        /// under the name generic math uses. Piecewise linear, so both second derivatives vanish.
        /// </summary>
        public readonly void Ieee754Remainder(in DDScalarSpan b, DDScalarSpan destination)
        {
            CheckCompatibility(this, b);
            CheckDestination(this, b, destination);

            double quotient = Math.Round(Value / b.Value, MidpointRounding.ToEven);
            double f = Math.IEEERemainder(Value, b.Value);

            Kernel.Binary<FalseTag, OneCoeff, ValueCoeff, ZeroCoeff, ZeroCoeff, ZeroCoeff>(
                AsReadOnlySpan(), b.AsReadOnlySpan(), f,
                default, new ValueCoeff(-quotient), default, default, default,
                destination.AsSpan(), _size, _order);
        }

        #endregion

        #region Value-Level Helpers

        /// <summary>Copies whichever operand wins the comparison, derivatives included.</summary>
        private readonly void SelectInto(bool takeThis, in DDScalarSpan other, DDScalarSpan destination)
        {
            if (takeThis) _data.CopyTo(destination._data);
            else other._data.CopyTo(destination._data);
        }

        /// <summary>Writes a derivative-free constant.</summary>
        private static void WriteConstant(double value, DDScalarSpan destination)
        {
            destination._data.Clear();
            destination._data[0] = value;
        }

        /// <summary>Selects by value and copies the winner into <paramref name="destination"/>.</summary>
        public readonly void Min(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);
            SelectInto(Value < other.Value, other, destination);
        }

        /// <summary>Selects by value and copies the winner into <paramref name="destination"/>.</summary>
        public readonly void Max(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);
            SelectInto(Value > other.Value, other, destination);
        }

        /// <summary>Selects by value and copies the winner into <paramref name="destination"/>.</summary>
        public readonly void MinMagnitude(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);
            SelectInto(Math.Abs(Value) < Math.Abs(other.Value), other, destination);
        }

        /// <summary>Selects by value and copies the winner into <paramref name="destination"/>.</summary>
        public readonly void MaxMagnitude(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);
            SelectInto(Math.Abs(Value) >= Math.Abs(other.Value), other, destination);
        }

        /// <summary>As <see cref="Min"/>, but a NaN operand loses against a number.</summary>
        public readonly void MinNumber(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);

            if (double.IsNaN(Value)) { other._data.CopyTo(destination._data); return; }
            if (double.IsNaN(other.Value)) { _data.CopyTo(destination._data); return; }

            SelectInto(Value < other.Value, other, destination);
        }

        /// <summary>As <see cref="Max"/>, but a NaN operand loses against a number.</summary>
        public readonly void MaxNumber(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);

            if (double.IsNaN(Value)) { other._data.CopyTo(destination._data); return; }
            if (double.IsNaN(other.Value)) { _data.CopyTo(destination._data); return; }

            SelectInto(Value > other.Value, other, destination);
        }

        /// <summary>As <see cref="MinMagnitude"/>, but a NaN operand loses against a number.</summary>
        public readonly void MinMagnitudeNumber(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);

            if (double.IsNaN(Value)) { other._data.CopyTo(destination._data); return; }
            if (double.IsNaN(other.Value)) { _data.CopyTo(destination._data); return; }

            SelectInto(Math.Abs(Value) < Math.Abs(other.Value), other, destination);
        }

        /// <summary>As <see cref="MaxMagnitude"/>, but a NaN operand loses against a number.</summary>
        public readonly void MaxMagnitudeNumber(in DDScalarSpan other, DDScalarSpan destination)
        {
            CheckCompatibility(this, other);
            CheckDestination(this, other, destination);

            if (double.IsNaN(Value)) { other._data.CopyTo(destination._data); return; }
            if (double.IsNaN(other.Value)) { _data.CopyTo(destination._data); return; }

            SelectInto(Math.Abs(Value) >= Math.Abs(other.Value), other, destination);
        }

        /// <summary>Selects this value, min or max, and copies the winner into <paramref name="destination"/>.</summary>
        public readonly void Clamp(in DDScalarSpan min, in DDScalarSpan max, DDScalarSpan destination)
        {
            CheckCompatibility(this, min);
            CheckCompatibility(this, max);
            CheckDestination(this, min, max, destination);
            if (min.Value > max.Value) throw new ArgumentException("min cannot be greater than max");

            if (Value < min.Value) min._data.CopyTo(destination._data);
            else if (Value > max.Value) max._data.CopyTo(destination._data);
            else _data.CopyTo(destination._data);
        }

        /// <summary>The sign as -1, 0 or 1. Piecewise constant, so the result carries no derivatives.</summary>
        public readonly void Sign(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Sign(Value), destination);
        }

        /// <summary>The magnitude of this value with the sign of <paramref name="sign"/>.</summary>
        public readonly void CopySign(in DDScalarSpan sign, DDScalarSpan destination)
        {
            CheckCompatibility(this, sign);
            CheckDestination(this, sign, destination);

            if (double.IsNegative(sign.Value) != (Value < 0.0)) Negate(destination);
            else _data.CopyTo(destination._data);
        }

        // Rounding is piecewise constant, so away from the break points the derivative is zero
        // and the result is a constant.

        public readonly void Round(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Round(Value), destination);
        }

        public readonly void Round(int digits, MidpointRounding mode, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Round(Value, digits, mode), destination);
        }

        public readonly void Round(MidpointRounding mode, DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Round(Value, mode), destination);
        }

        public readonly void Floor(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Floor(Value), destination);
        }

        public readonly void Ceiling(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Ceiling(Value), destination);
        }

        public readonly void Truncate(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            WriteConstant(Math.Truncate(Value), destination);
        }

        /// <summary>
        /// The neighbouring representable value. Within a binade the step is a constant, so this is
        /// <c>x + c</c> and the derivatives survive it — unlike the piecewise-constant rounding above.
        /// </summary>
        public readonly void BitIncrement(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            _data.CopyTo(destination._data);
            destination._data[0] = double.BitIncrement(Value);
        }

        /// <summary>The next representable value below, keeping the derivatives.</summary>
        public readonly void BitDecrement(DDScalarSpan destination)
        {
            CheckDestination(this, destination);
            _data.CopyTo(destination._data);
            destination._data[0] = double.BitDecrement(Value);
        }

        /// <summary>
        /// Multiplies by <c>2^n</c>. Linear, so every coefficient scales with it; using ScaleB per
        /// coefficient keeps the scaling exact.
        /// </summary>
        public readonly void ScaleB(int n, DDScalarSpan destination)
        {
            CheckDestination(this, destination);

            for (int i = 0; i < _data.Length; i++) destination._data[i] = Math.ScaleB(_data[i], n);
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