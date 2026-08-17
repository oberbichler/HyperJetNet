// Compiled against the produced NuGet package rather than the project, so that "the package
// builds" and "the package is usable" stop being the same claim.
//
// The part nothing else exercises is build/HyperJet.targets: the generator's `global using`
// directives only apply inside HyperJet's own compilation, so the unqualified DDScalar2
// spelling a consumer sees comes from that file. If it stopped being packed, or listed
// the wrong types, every other check in this repository would still pass.

using static HyperJet.HyperJetMath;

int failures = 0;

void Check(string what, double expected, double actual)
{
    bool ok = Math.Abs(expected - actual) < 1e-9;
    Console.WriteLine($"{(ok ? "ok  " : "FAIL")}  {what}: expected {expected}, got {actual}");
    if (!ok) failures++;
}

// `using HyperJet` itself comes from the targets file.
Console.WriteLine($"HyperJet {typeof(DDScalar).Assembly.GetName().Version}");

// Second-order family, through the unqualified alias.
var (x, y) = DDScalar2.Variables(3.0, 6.0);
DDScalar2 f = (x * y) / (x - y);
Check("DDScalar2 value", -6.0, f.Value);
Check("DDScalar2 df/dx", -4.0, f.G(0));
Check("DDScalar2 d2f/dx2", -8.0 / 3.0, f.H(0, 0));

// The free-function facade, which a consumer imports themselves.
Check("Sin via the facade", Math.Sin(1.0), Sin(DDScalar2.Variable(0, 1.0)).Value);

// The dynamic model and the kernel.
DDScalar[] v = DDScalar.Variables(new[] { 1.5, 2.5 });
Check("DDScalar value", 3.75, (v[0] * v[1]).Value);
Check("Kernel data length", 6.0, HyperJet.Kernel.GetDataLength(2, 2));

// The local Taylor model, which was missing from the package until recently.
Check("Evaluate", f.Evaluate(0.0, 0.0), f.Value);

Console.WriteLine(failures == 0
    ? "package smoke test passed"
    : $"package smoke test: {failures} check(s) failed");

return failures == 0 ? 0 : 1;
