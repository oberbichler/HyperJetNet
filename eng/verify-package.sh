#!/usr/bin/env bash
#
# Compiles and runs a fresh project against the produced package.
#
# The consumer project is scaffolded here rather than committed, because `dotnet build` at the
# repository root discovers projects recursively: a committed consumer would join the ordinary
# build and fail there, since the package it needs does not exist yet at that point.
#
# Usage: eng/verify-package.sh [artifacts-directory]

set -euo pipefail

artifacts=${1:-artifacts}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

package=$(find "$artifacts" -maxdepth 1 -name 'HyperJet.*.nupkg' | sort | head -1)
if [ -z "$package" ]; then
  echo "no HyperJet package in $artifacts"
  exit 1
fi
feed=$(cd "$(dirname "$package")" && pwd)
version=$(basename "$package" .nupkg)
version=${version#HyperJet.}

echo "verifying $(basename "$package")"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A private package cache. Without it NuGet reuses an already-extracted copy of the same id and
# version from ~/.nuget/packages, so a rebuilt -- or broken -- package is never actually read and
# the check silently passes on stale content.
export NUGET_PACKAGES="$work/packages"

cd "$work"
dotnet new console -o consumer >/dev/null

# Only the freshly built package is visible, so a restore cannot quietly fall back to a version
# that is already on nuget.org and report success for the wrong artifact.
cat > nuget.config <<XML
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="local" value="$feed" />
  </packageSources>
</configuration>
XML

cp "$here/package-smoke-test/Program.cs" consumer/Program.cs

cd consumer
dotnet add package HyperJet --version "$version" >/dev/null
dotnet run
