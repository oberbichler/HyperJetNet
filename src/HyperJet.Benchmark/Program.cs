using BenchmarkDotNet.Running;
using System;

namespace HyperJet.Benchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=========================================================");
            Console.WriteLine("    HyperJet.Net - Automatic Differentiation Benchmark   ");
            Console.WriteLine("=========================================================");
            Console.WriteLine($".NET Runtime Version: {Environment.Version}");
            Console.WriteLine($"Is 64-bit Process: {Environment.Is64BitProcess}");
            Console.WriteLine();

            // Run BenchmarkDotNet
            BenchmarkRunner.Run<AdBenchmark>();
        }
    }
}