# HyperJet

[![.NET 10](https://img.shields.io/badge/.NET-10.0-blue.svg)](https://dotnet.microsoft.com/)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](LICENSE)

**HyperJet** is a highly efficient, modern C# library for **Automatic Differentiation (AD)** of first and second order (gradients and Hessian matrices). Engineered from the ground up for **.NET 10** and modern CPU architectures, it combines maximum type safety with extreme mathematical performance through SIMD hardware acceleration, source generators, and zero-allocation stack structures.

---

## Key Features

### 🚀 High-Performance & Hardware Acceleration

- **Explicit SIMD Vectorization**: Hand-optimized, low-level mathematical kernels leverage the latest CPU instruction sets via .NET Hardware Intrinsics (`Vector512` for AVX-512, `Vector256` for AVX2).
- **Native Apple Silicon & ARM64 Support**: Native `Vector128` integration enables **Apple Silicon (M1/M2/M3/M4)** and other ARM64 hardware to benefit fully from hardware-vectorized Neon instructions.

### 🛠️ Three Tailored Computational Models

1. **Static Compile-Time Structs (`DDScalar1` to `DDScalar15`)**:
   - Fully generated via **C# Source Generators** for up to 15 variables.
   - **100% Stack-allocated** (zero heap overhead, zero garbage collection impact).
   - Seamlessly integrated with the **.NET Generic Math System** (`IFloatingPoint<T>`).
2. **Ref Structs for Zero-Allocation (`DDScalarSpan`)**:
   - Allows dynamic, runtime-determined variable counts with zero heap allocations.
   - Operates directly on stack-allocated buffers (`stackalloc double[]`).
   - Full mathematical parity via high-performance zero-allocation instance methods (`Sin`, `Cos`, `Exp`, `Sinh`, etc.).
3. **Dynamic Heap-Allocated Structs (`DDScalar`)**:
   - Maximum flexibility for dynamic expression evaluation with runtime-determined sizes and orders.

### 💎 Premium Usability & API

- **Tuple Deconstruction**: Seamlessly unpack variable arrays, spans, or ReadOnlySpans into local variables for up to 15 dimensions:
  ```csharp
  var (x, y, z) = DDScalar.Variables(new double[] { 1.5, 3.0, 4.5 });
  ```
- **Easy Vector & Matrix Exports**: Direct extraction of gradient vectors or the full Hessian matrix via `GetGradient()` and `GetHessian()` for immediate downstream integration (e.g., with Math.NET).
- **Generic Geometric Vector `Vector3D<T>`**:
  - Out-of-the-box generic 3D vector structure (`where T : IFloatingPoint<T>, IRootFunctions<T>`).
  - Supports vector addition, scaling, **dot product**, **cross product**, **length**, and **normalization**.
  - Allows mixing constants and active variables seamlessly through implicit type conversion from `double`/`float` to `DDScalar{n}`.

---

## Installation

Simply install the **HyperJet** NuGet package into your .NET 10 project:

```bash
dotnet add package HyperJet
```

---

## Code Examples

### 1. Static Compile-Time AD with `DDScalar2`

Ideal when the number of variables (here $n=2$) is known at compile-time. Expressions are written naturally using standard mathematical syntax:

```csharp
using HyperJet;
using static HyperJet.HyperJetMath;

// Initialize variables (x = 3.0, y = 6.0)
var (x, y) = DDScalar2.Variables(3.0, 6.0);

// Mathematical function f(x, y) = (x * y) / (x - y)
DDScalar2 f = (x * y) / (x - y);

Console.WriteLine($"Function value f(x, y): {f.Value}"); // -6.0
Console.WriteLine($"First derivative df/dx: {f.G(0)}");   // -4.0
Console.WriteLine($"Second derivative d²f/dx²: {f.H(0, 0)}"); // -2.666666667
```

### 2. Zero-Allocation AD on the Stack with `DDScalarSpan`

Ideal for maximum performance with runtime-dynamic variable counts (0 bytes of heap allocation):

```csharp
using System;
using HyperJet;

int size = 2; // Dynamic size determined at runtime
int dataLength = Kernel.GetDataLength(size, order: 2);

// Allocate buffers on stack (0 heap allocations)
Span<double> xBuffer = stackalloc double[dataLength];
Span<double> yBuffer = stackalloc double[dataLength];
Span<double> resultBuffer = stackalloc double[dataLength];

var x = DDScalarSpan.Variable(xBuffer, 0, 3.0, size, order: 2);
var y = DDScalarSpan.Variable(yBuffer, 1, 6.0, size, order: 2);
var result = new DDScalarSpan(resultBuffer, size, order: 2);

// Evaluate via zero-allocation instance methods
x.Sin(result); // result = sin(x)

Console.WriteLine($"Value: {result.Value}"); // Math.Sin(3.0)
Console.WriteLine($"df/dx: {result.G(0)}"); // Math.Cos(3.0)
```

### 3. Generic Physics & Geometrical Operations with `Vector3D<T>`

Written once, this generic code runs with standard double-precision floats for physical simulation, or with DDScalar for automatic differentiation:

```csharp
using HyperJet;
using System.Numerics;

public static class Physics
{
    // Compute torque Tau = r x F
    public static Vector3D<T> CalculateTorque<T>(Vector3D<T> r, Vector3D<T> F)
        where T : IFloatingPoint<T>, IRootFunctions<T>
    {
        return r.Cross(F); // Cross product
    }
}

// DOWNSTREAM APPLICATION WITH DUAL NUMBERS:
var (x, y, z) = DDScalar3<double>.Variables(2.0, 0.0, 0.0);
var r = new Vector3D<DDScalar3<double>>(x, y, z);
var F = new Vector3D<DDScalar3<double>>(0.0, 10.0, 0.0); // 10.0 converts implicitly!

// Compute torque
Vector3D<DDScalar3<double>> torque = Physics.CalculateTorque(r, F);
DDScalar3<double> torqueZ = torque.Z;

Console.WriteLine($"Lever-arm sensitivity on Z-torque: {torqueZ.G(1)}"); // -10.0
```

---

## Performance & Benchmarks

Benchmark measurements on **Apple M1 Pro (macOS)** demonstrate the outstanding efficiency of optimizations under .NET 10:

```
BenchmarkDotNet v0.14.0, macOS 26.5
Apple M1 Pro, 1 CPU, 10 logical and 10 physical cores
IsHardwareAccelerated (ARM64 Neon / AdvSIMD) = True
```

| Method                          | Size ($n$) |      Mean | Gen 0 / 1000 ops | Heap Allocation |
| ------------------------------- | :--------: | --------: | ---------------: | --------------: |
| `StaticDouble_DDScalar2`        |     2      |  56.29 ns |                - |         **0 B** |
| `DynamicHeap_DDScalar`          |     2      |  53.41 ns |           0.0573 |           360 B |
| `DynamicStack_DDScalarSpan`     |     2      |  39.58 ns |                - |         **0 B** |
| `DynamicHeap_ScalarOnly_NoSIMD` |     2      |  40.29 ns |           0.0573 |           360 B |
|                                 |            |           |                  |                 |
| `DynamicHeap_DDScalar` (SIMD)   |     10     | 312.97 ns |           0.4396 |          2760 B |
| `DynamicStack_DDScalarSpan`     |     10     | 246.11 ns |                - |         **0 B** |
| `DynamicHeap_ScalarOnly_NoSIMD` |     10     | 332.85 ns |           0.4396 |          2760 B |

### Key Insights:

1. **SIMD Vectorization**: At larger scales ($n=10$), native `Vector128` vectorization on Apple Silicon reduces execution time by **~6.3%** (`312 ns` vs. `332 ns` without SIMD).
2. **Zero-Allocation Stack Power**: `DDScalarSpan` is **~21% to 26% faster** than its heap-allocated counterpart due to direct reference passing, while completely avoiding Garbage Collection pressure.

---

## License

This project is licensed under the **ISC License** – see the [LICENSE](LICENSE) file for details.
