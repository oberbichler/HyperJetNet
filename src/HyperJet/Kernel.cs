using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace HyperJet
{
    /// <summary>
    /// Highly optimized, low-level mathematical kernels for automatic differentiation propagation.
    /// Supports 1st and 2nd order derivatives for unary, binary, and ternary functions.
    /// </summary>
    public static class Kernel
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetDataLength(int size, int order)
        {
            return order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetSizeFromDataLength(int length, int order)
        {
            if (order == 1)
            {
                return length - 1;
            }

            int s = (int)(Math.Sqrt(1 + 8 * length) - 3) / 2;
            if (s < 0 || GetDataLength(s, order) != length)
            {
                throw new ArgumentException($"Invalid data length {length} for 2nd order dual numbers.");
            }
            return s;
        }

        #region Double Overloads (Backward Compatibility)

        /// <summary>
        /// Extracts the scalar factor of a coefficient so it can be broadcast into a SIMD vector.
        /// <para>
        /// Coefficients are linear (<c>Multiply(v) == c * v</c>), so the factor is simply
        /// <c>Multiply(1.0)</c>. Deriving it from the coefficient itself — rather than from a
        /// hard-coded table of the known tag types — keeps the vectorized path and the scalar
        /// remainder loop in agreement for <em>every</em> <see cref="ICoeff"/> implementation.
        /// For the built-in tags the call inlines to a constant, so no codegen is lost.
        /// </para>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double GetCoeffValue<TCoeff>(in TCoeff coeff) where TCoeff : struct, ICoeff
        {
            return coeff.Multiply(1.0);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Unary<TIncrement, TDa, TDaa>(
            ReadOnlySpan<double> a,
            double f,
            TDa da,
            TDaa daa,
            Span<double> r,
            int size,
            int order)
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff
            where TDaa : struct, ICoeff
        {
            r[0] = f;

            if (order < 1)
            {
                return;
            }

            int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;

            if (typeof(TDa) == typeof(ZeroCoeff))
            {
                // Nothing to propagate, but in assign mode the kernel still owns every slot.
                if (typeof(TIncrement) != typeof(TrueTag))
                {
                    r.Slice(1, n - 1).Clear();
                }

                PropagateSecondOrder<TDaa>(a, daa, r, size, order);
                return;
            }

            double daVal = GetCoeffValue(da);
            int i = 1;

            if (n >= 9 && Vector512.IsHardwareAccelerated)
            {
                var daVec = Vector512.Create(daVal);
                ref double aRef = ref Unsafe.AsRef(in a[0]);
                ref double rRef = ref MemoryMarshal.GetReference(r);
                int limit = n - (n - 1) % 8;
                for (; i < limit; i += 8)
                {
                    var va = Vector512.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                    var term = va * daVec;
                    if (typeof(TIncrement) == typeof(TrueTag))
                    {
                        var vr = Vector512.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                        (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                    else
                    {
                        term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                }
            }
            else if (n >= 5 && Vector256.IsHardwareAccelerated)
            {
                var daVec = Vector256.Create(daVal);
                ref double aRef = ref Unsafe.AsRef(in a[0]);
                ref double rRef = ref MemoryMarshal.GetReference(r);
                int limit = n - (n - 1) % 4;
                for (; i < limit; i += 4)
                {
                    var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                    var term = va * daVec;
                    if (typeof(TIncrement) == typeof(TrueTag))
                    {
                        var vr = Vector256.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                        (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                    else
                    {
                        term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                }
            }
            else if (n >= 3 && Vector128.IsHardwareAccelerated)
            {
                var daVec = Vector128.Create(daVal);
                ref double aRef = ref Unsafe.AsRef(in a[0]);
                ref double rRef = ref MemoryMarshal.GetReference(r);
                int limit = n - (n - 1) % 2;
                for (; i < limit; i += 2)
                {
                    var va = Vector128.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                    var term = va * daVec;
                    if (typeof(TIncrement) == typeof(TrueTag))
                    {
                        var vr = Vector128.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                        (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                    else
                    {
                        term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                    }
                }
            }

            // Remainder loop
            for (; i < n; i++)
            {
                double term = da.Multiply(a[i]);
                if (typeof(TIncrement) == typeof(TrueTag))
                {
                    r[i] += term;
                }
                else
                {
                    r[i] = term;
                }
            }

            PropagateSecondOrder<TDaa>(a, daa, r, size, order);
        }

        /// <summary>
        /// Adds the second-order chain-rule contribution <c>daa * grad(a) grad(a)^T</c> to the
        /// Hessian block. This term is independent of the first-derivative coefficient, so it must
        /// run even when the first derivative vanishes.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PropagateSecondOrder<TDaa>(
            ReadOnlySpan<double> a,
            TDaa daa,
            Span<double> r,
            int size,
            int order)
            where TDaa : struct, ICoeff
        {
            if (order < 2 || typeof(TDaa) == typeof(ZeroCoeff))
            {
                return;
            }

            int k = 1 + size;
            for (int j = 0; j < size; j++)
            {
                double ca = daa.Multiply(a[1 + j]);
                for (int m = j; m < size; m++)
                {
                    r[k++] += ca * a[1 + m];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Binary<TIncrement, TDa, TDb, TDaa, TDab, TDbb>(
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            double f,
            TDa da,
            TDb db,
            TDaa daa,
            TDab dab,
            TDbb dbb,
            Span<double> r,
            int size,
            int order)
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff
            where TDb : struct, ICoeff
            where TDaa : struct, ICoeff
            where TDab : struct, ICoeff
            where TDbb : struct, ICoeff
        {
            r[0] = f;

            if (order >= 1)
            {
                bool isDaZero = typeof(TDa) == typeof(ZeroCoeff);
                bool isDbZero = typeof(TDb) == typeof(ZeroCoeff);

                if (isDaZero && isDbZero)
                {
                    // Nothing to propagate, but in assign mode the kernel still owns every slot.
                    if (typeof(TIncrement) != typeof(TrueTag))
                    {
                        int cleared = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
                        r.Slice(1, cleared - 1).Clear();
                    }
                }
                else
                {
                    int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
                    double daVal = GetCoeffValue(da);
                    double dbVal = GetCoeffValue(db);
                    int i = 1;

                    if (n >= 9 && Vector512.IsHardwareAccelerated)
                    {
                        var daVec = Vector512.Create(daVal);
                        var dbVec = Vector512.Create(dbVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 8;
                        for (; i < limit; i += 8)
                        {
                            var va = Vector512.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector512.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var term = va * daVec + vb * dbVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector512.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }
                    else if (n >= 5 && Vector256.IsHardwareAccelerated)
                    {
                        var daVec = Vector256.Create(daVal);
                        var dbVec = Vector256.Create(dbVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 4;
                        for (; i < limit; i += 4)
                        {
                            var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var term = va * daVec + vb * dbVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector256.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }
                    else if (n >= 3 && Vector128.IsHardwareAccelerated)
                    {
                        var daVec = Vector128.Create(daVal);
                        var dbVec = Vector128.Create(dbVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 2;
                        for (; i < limit; i += 2)
                        {
                            var va = Vector128.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector128.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var term = va * daVec + vb * dbVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector128.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }

                    // Remainder loop
                    for (; i < n; i++)
                    {
                        double term = da.Multiply(a[i]) + db.Multiply(b[i]);
                        if (typeof(TIncrement) == typeof(TrueTag))
                        {
                            r[i] += term;
                        }
                        else
                        {
                            r[i] = term;
                        }
                    }
                }
            }

            if (order >= 2)
            {
                bool isDaaZero = typeof(TDaa) == typeof(ZeroCoeff);
                bool isDabZero = typeof(TDab) == typeof(ZeroCoeff);
                bool isDbbZero = typeof(TDbb) == typeof(ZeroCoeff);

                if (!isDaaZero || !isDabZero || !isDbbZero)
                {
                    int k = 1 + size;
                    for (int i = 0; i < size; i++)
                    {
                        double ai = a[1 + i];
                        double bi = b[1 + i];
                        double ca = daa.Multiply(ai) + dab.Multiply(bi);
                        double cb = dab.Multiply(ai) + dbb.Multiply(bi);

                        for (int j = i; j < size; j++)
                        {
                            r[k++] += ca * a[1 + j] + cb * b[1 + j];
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Ternary<TIncrement, TDa, TDb, TDc, TDaa, TDab, TDac, TDbb, TDbc, TDcc>(
            ReadOnlySpan<double> a,
            ReadOnlySpan<double> b,
            ReadOnlySpan<double> c,
            double f,
            TDa da,
            TDb db,
            TDc dc,
            TDaa daa,
            TDab dab,
            TDac dac,
            TDbb dbb,
            TDbc dbc,
            TDcc dcc,
            Span<double> r,
            int size,
            int order)
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff
            where TDb : struct, ICoeff
            where TDc : struct, ICoeff
            where TDaa : struct, ICoeff
            where TDab : struct, ICoeff
            where TDac : struct, ICoeff
            where TDbb : struct, ICoeff
            where TDbc : struct, ICoeff
            where TDcc : struct, ICoeff
        {
            r[0] = f;

            if (order >= 1)
            {
                bool isDaZero = typeof(TDa) == typeof(ZeroCoeff);
                bool isDbZero = typeof(TDb) == typeof(ZeroCoeff);
                bool isDcZero = typeof(TDc) == typeof(ZeroCoeff);

                if (isDaZero && isDbZero && isDcZero)
                {
                    // Nothing to propagate, but in assign mode the kernel still owns every slot.
                    if (typeof(TIncrement) != typeof(TrueTag))
                    {
                        int cleared = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
                        r.Slice(1, cleared - 1).Clear();
                    }
                }
                else
                {
                    int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;
                    double daVal = GetCoeffValue(da);
                    double dbVal = GetCoeffValue(db);
                    double dcVal = GetCoeffValue(dc);
                    int i = 1;

                    if (n >= 9 && Vector512.IsHardwareAccelerated)
                    {
                        var daVec = Vector512.Create(daVal);
                        var dbVec = Vector512.Create(dbVal);
                        var dcVec = Vector512.Create(dcVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double cRef = ref Unsafe.AsRef(in c[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 8;
                        for (; i < limit; i += 8)
                        {
                            var va = Vector512.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector512.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var vc = Vector512.LoadUnsafe(ref Unsafe.Add(ref cRef, i));
                            var term = va * daVec + vb * dbVec + vc * dcVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector512.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }
                    else if (n >= 5 && Vector256.IsHardwareAccelerated)
                    {
                        var daVec = Vector256.Create(daVal);
                        var dbVec = Vector256.Create(dbVal);
                        var dcVec = Vector256.Create(dcVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double cRef = ref Unsafe.AsRef(in c[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 4;
                        for (; i < limit; i += 4)
                        {
                            var va = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var vc = Vector256.LoadUnsafe(ref Unsafe.Add(ref cRef, i));
                            var term = va * daVec + vb * dbVec + vc * dcVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector256.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }
                    else if (n >= 3 && Vector128.IsHardwareAccelerated)
                    {
                        var daVec = Vector128.Create(daVal);
                        var dbVec = Vector128.Create(dbVal);
                        var dcVec = Vector128.Create(dcVal);
                        ref double aRef = ref Unsafe.AsRef(in a[0]);
                        ref double bRef = ref Unsafe.AsRef(in b[0]);
                        ref double cRef = ref Unsafe.AsRef(in c[0]);
                        ref double rRef = ref MemoryMarshal.GetReference(r);
                        int limit = n - (n - 1) % 2;
                        for (; i < limit; i += 2)
                        {
                            var va = Vector128.LoadUnsafe(ref Unsafe.Add(ref aRef, i));
                            var vb = Vector128.LoadUnsafe(ref Unsafe.Add(ref bRef, i));
                            var vc = Vector128.LoadUnsafe(ref Unsafe.Add(ref cRef, i));
                            var term = va * daVec + vb * dbVec + vc * dcVec;
                            if (typeof(TIncrement) == typeof(TrueTag))
                            {
                                var vr = Vector128.LoadUnsafe(ref Unsafe.Add(ref rRef, i));
                                (vr + term).StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                            else
                            {
                                term.StoreUnsafe(ref Unsafe.Add(ref rRef, i));
                            }
                        }
                    }

                    // Remainder loop
                    for (; i < n; i++)
                    {
                        double term = da.Multiply(a[i]) + db.Multiply(b[i]) + dc.Multiply(c[i]);
                        if (typeof(TIncrement) == typeof(TrueTag))
                        {
                            r[i] += term;
                        }
                        else
                        {
                            r[i] = term;
                        }
                    }
                }
            }

            if (order >= 2)
            {
                bool isDaaZero = typeof(TDaa) == typeof(ZeroCoeff);
                bool isDabZero = typeof(TDab) == typeof(ZeroCoeff);
                bool isDacZero = typeof(TDac) == typeof(ZeroCoeff);
                bool isDbbZero = typeof(TDbb) == typeof(ZeroCoeff);
                bool isDbcZero = typeof(TDbc) == typeof(ZeroCoeff);
                bool isDccZero = typeof(TDcc) == typeof(ZeroCoeff);

                if (!isDaaZero || !isDabZero || !isDacZero || !isDbbZero || !isDbcZero || !isDccZero)
                {
                    int k = 1 + size;
                    for (int i = 0; i < size; i++)
                    {
                        double ai = a[1 + i];
                        double bi = b[1 + i];
                        double ci = c[1 + i];
                        double ca = daa.Multiply(ai) + dab.Multiply(bi) + dac.Multiply(ci);
                        double cb = dab.Multiply(ai) + dbb.Multiply(bi) + dbc.Multiply(ci);
                        double cc = dac.Multiply(ai) + dbc.Multiply(bi) + dcc.Multiply(ci);

                        for (int j = i; j < size; j++)
                        {
                            r[k++] += ca * a[1 + j] + cb * b[1 + j] + cc * c[1 + j];
                        }
                    }
                }
            }
        }

        #endregion

        #region Generic Overloads

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Unary<T, TIncrement, TDa, TDaa>(
            ReadOnlySpan<T> a,
            T f,
            TDa da,
            TDaa daa,
            Span<T> r,
            int size,
            int order)
            where T : IFloatingPoint<T>
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff<T>
            where TDaa : struct, ICoeff<T>
        {
            r[0] = f;

            if (order < 1)
            {
                return;
            }

            int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;

            if (IsZeroCoeff<TDa>())
            {
                // Nothing to propagate, but in assign mode the kernel still owns every slot.
                if (typeof(TIncrement) != typeof(TrueTag))
                {
                    for (int i = 1; i < n; i++) r[i] = T.Zero;
                }
            }
            else
            {
                for (int i = 1; i < n; i++)
                {
                    T term = da.Multiply(a[i]);
                    if (typeof(TIncrement) == typeof(TrueTag))
                    {
                        r[i] += term;
                    }
                    else
                    {
                        r[i] = term;
                    }
                }
            }

            if (order < 2 || IsZeroCoeff<TDaa>())
            {
                return;
            }

            int k = 1 + size;
            for (int i = 0; i < size; i++)
            {
                T ca = daa.Multiply(a[1 + i]);
                for (int j = i; j < size; j++)
                {
                    r[k++] += ca * a[1 + j];
                }
            }
        }

        /// <summary>Whether <typeparamref name="TCoeff"/> is the generic zero tag.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsZeroCoeff<TCoeff>() =>
            typeof(TCoeff).IsGenericType && typeof(TCoeff).GetGenericTypeDefinition() == typeof(ZeroCoeff<>);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Binary<T, TIncrement, TDa, TDb, TDaa, TDab, TDbb>(
            ReadOnlySpan<T> a,
            ReadOnlySpan<T> b,
            T f,
            TDa da,
            TDb db,
            TDaa daa,
            TDab dab,
            TDbb dbb,
            Span<T> r,
            int size,
            int order)
            where T : IFloatingPoint<T>
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff<T>
            where TDb : struct, ICoeff<T>
            where TDaa : struct, ICoeff<T>
            where TDab : struct, ICoeff<T>
            where TDbb : struct, ICoeff<T>
        {
            r[0] = f;

            if (order >= 1)
            {
                int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;

                if (IsZeroCoeff<TDa>() && IsZeroCoeff<TDb>())
                {
                    // Nothing to propagate, but in assign mode the kernel still owns every slot.
                    if (typeof(TIncrement) != typeof(TrueTag))
                    {
                        for (int i = 1; i < n; i++) r[i] = T.Zero;
                    }
                }
                else
                {
                    for (int i = 1; i < n; i++)
                    {
                        T term = da.Multiply(a[i]) + db.Multiply(b[i]);
                        if (typeof(TIncrement) == typeof(TrueTag))
                        {
                            r[i] += term;
                        }
                        else
                        {
                            r[i] = term;
                        }
                    }
                }
            }

            if (order >= 2)
            {
                bool isDaaZero = IsZeroCoeff<TDaa>();
                bool isDabZero = IsZeroCoeff<TDab>();
                bool isDbbZero = IsZeroCoeff<TDbb>();

                if (!isDaaZero || !isDabZero || !isDbbZero)
                {
                    int k = 1 + size;
                    for (int i = 0; i < size; i++)
                    {
                        T ai = a[1 + i];
                        T bi = b[1 + i];
                        T ca = daa.Multiply(ai) + dab.Multiply(bi);
                        T cb = dab.Multiply(ai) + dbb.Multiply(bi);

                        for (int j = i; j < size; j++)
                        {
                            r[k++] += ca * a[1 + j] + cb * b[1 + j];
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Ternary<T, TIncrement, TDa, TDb, TDc, TDaa, TDab, TDac, TDbb, TDbc, TDcc>(
            ReadOnlySpan<T> a,
            ReadOnlySpan<T> b,
            ReadOnlySpan<T> c,
            T f,
            TDa da,
            TDb db,
            TDc dc,
            TDaa daa,
            TDab dab,
            TDac dac,
            TDbb dbb,
            TDbc dbc,
            TDcc dcc,
            Span<T> r,
            int size,
            int order)
            where T : IFloatingPoint<T>
            where TIncrement : struct, ITag
            where TDa : struct, ICoeff<T>
            where TDb : struct, ICoeff<T>
            where TDc : struct, ICoeff<T>
            where TDaa : struct, ICoeff<T>
            where TDab : struct, ICoeff<T>
            where TDac : struct, ICoeff<T>
            where TDbb : struct, ICoeff<T>
            where TDbc : struct, ICoeff<T>
            where TDcc : struct, ICoeff<T>
        {
            r[0] = f;

            if (order >= 1)
            {
                int n = order == 1 ? 1 + size : (size + 1) * (size + 2) / 2;

                if (IsZeroCoeff<TDa>() && IsZeroCoeff<TDb>() && IsZeroCoeff<TDc>())
                {
                    // Nothing to propagate, but in assign mode the kernel still owns every slot.
                    if (typeof(TIncrement) != typeof(TrueTag))
                    {
                        for (int i = 1; i < n; i++) r[i] = T.Zero;
                    }
                }
                else
                {
                    for (int i = 1; i < n; i++)
                    {
                        T term = da.Multiply(a[i]) + db.Multiply(b[i]) + dc.Multiply(c[i]);
                        if (typeof(TIncrement) == typeof(TrueTag))
                        {
                            r[i] += term;
                        }
                        else
                        {
                            r[i] = term;
                        }
                    }
                }
            }

            if (order >= 2)
            {
                bool isDaaZero = IsZeroCoeff<TDaa>();
                bool isDabZero = IsZeroCoeff<TDab>();
                bool isDacZero = IsZeroCoeff<TDac>();
                bool isDbbZero = IsZeroCoeff<TDbb>();
                bool isDbcZero = IsZeroCoeff<TDbc>();
                bool isDccZero = IsZeroCoeff<TDcc>();

                if (!isDaaZero || !isDabZero || !isDacZero || !isDbbZero || !isDbcZero || !isDccZero)
                {
                    int k = 1 + size;
                    for (int i = 0; i < size; i++)
                    {
                        T ai = a[1 + i];
                        T bi = b[1 + i];
                        T ci = c[1 + i];
                        T ca = daa.Multiply(ai) + dab.Multiply(bi) + dac.Multiply(ci);
                        T cb = dab.Multiply(ai) + dbb.Multiply(bi) + dbc.Multiply(ci);
                        T cc = dac.Multiply(ai) + dbc.Multiply(bi) + dcc.Multiply(ci);

                        for (int j = i; j < size; j++)
                        {
                            r[k++] += ca * a[1 + j] + cb * b[1 + j] + cc * c[1 + j];
                        }
                    }
                }
            }
        }

        #endregion
    }
}