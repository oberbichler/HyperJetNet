using System.Numerics;
using System.Runtime.CompilerServices;

namespace HyperJet
{
    /// <summary>
    /// Represents a coefficient multiplier used in derivative propagation.
    /// This allows the JIT compiler to optimize multiplication operations (e.g., eliminating multiplications by zero or one).
    /// </summary>
    public interface ICoeff
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        double Multiply(double val);
    }

    /// <summary>
    /// A coefficient tag representing zero. Multiplications are optimized away.
    /// </summary>
    public struct ZeroCoeff : ICoeff
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Multiply(double val) => 0.0;
    }

    /// <summary>
    /// A coefficient tag representing one. Multiplications are identity operations.
    /// </summary>
    public struct OneCoeff : ICoeff
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Multiply(double val) => val;
    }

    /// <summary>
    /// A coefficient tag representing minus one.
    /// </summary>
    public struct MinusOneCoeff : ICoeff
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Multiply(double val) => -val;
    }

    /// <summary>
    /// A general double coefficient value.
    /// </summary>
    public readonly struct ValueCoeff : ICoeff
    {
        public readonly double Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ValueCoeff(double value)
        {
            Value = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Multiply(double val) => Value * val;
    }

    /// <summary>
    /// Represents a generic coefficient multiplier.
    /// </summary>
    public interface ICoeff<T> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        T Multiply(T val);
    }

    /// <summary>
    /// A coefficient tag representing zero. Multiplications are optimized away.
    /// </summary>
    public struct ZeroCoeff<T> : ICoeff<T> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Multiply(T val) => T.Zero;
    }

    /// <summary>
    /// A coefficient tag representing one. Multiplications are identity operations.
    /// </summary>
    public struct OneCoeff<T> : ICoeff<T> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Multiply(T val) => val;
    }

    /// <summary>
    /// A coefficient tag representing minus one.
    /// </summary>
    public struct MinusOneCoeff<T> : ICoeff<T> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Multiply(T val) => -val;
    }

    /// <summary>
    /// A general coefficient value.
    /// </summary>
    public readonly struct ValueCoeff<T> : ICoeff<T> where T : IFloatingPoint<T>
    {
        public readonly T Value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ValueCoeff(T value)
        {
            Value = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T Multiply(T val) => Value * val;
    }

    /// <summary>
    /// Tag interface to control whether unary/binary/ternary operations assign or increment.
    /// </summary>
    public interface ITag { }

    /// <summary>
    /// Tag indicating values should be incremented/added (accumulated) to the destination.
    /// </summary>
    public struct TrueTag : ITag { }

    /// <summary>
    /// Tag indicating values should be directly assigned (overwritten) to the destination.
    /// </summary>
    public struct FalseTag : ITag { }
}