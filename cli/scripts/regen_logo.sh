#!/usr/bin/env bash
# Regenerate cli/logo.py from Coralbricks/webapp/public/logo-icon.svg.
#
# Requires: rsvg-convert (librsvg), chafa, perl, python3.
#   brew install librsvg chafa
#
# Run from anywhere. Writes cli/logo.py in place.
#
# Flags worth keeping aligned:
# - --colors full: truecolor so the coral fill (#d44027) round-trips exactly
#                  instead of being quantized toward orange by the 256 palette.
# - --font-ratio=0.42: most macOS terminal fonts (Menlo, SF Mono, Monaco)
#                     render cells closer to 2.3:1 (h:w) than chafa's default
#                     2:1 assumption. Without this, the round magnifying-glass
#                     rasterizes as a tall oval.
# - --size 24x10: paired with the font-ratio, this gives a near-square
#                 display on typical macOS terminals while keeping the art
#                 narrow enough to sit next to the wordmark.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cli_dir="$(dirname "$here")"
repo_root="$(cd "$cli_dir/../.." && pwd)"

svg="$repo_root/Coralbricks/webapp/public/logo-icon.svg"
out="$cli_dir/logo.py"
png="$(mktemp -t coralbricks-logo.XXXXXX.png)"
trap 'rm -f "$png"' EXIT

if [[ ! -f "$svg" ]]; then
  echo "error: logo svg not found at $svg" >&2
  exit 1
fi

rsvg-convert -w 400 "$svg" -o "$png"

# chafa emits cursor hide/show escapes (\e[?25l, \e[?25h) around the art;
# strip them so the rendered output is pure color + glyph content.
ansi="$(
  chafa \
    --animate off \
    --symbols block \
    --size 24x10 \
    --font-ratio=0.42 \
    --colors full \
    --format symbols \
    "$png" \
    | perl -pe 's/\e\[\?25[hl]//g'
)"

ANSI="$ansi" python3 - "$out" <<'PY'
import os, sys, pathlib
out = pathlib.Path(sys.argv[1])
ansi = os.environ["ANSI"]
lines = ansi.splitlines()
py_lines = []
for line in lines:
    escaped = (
        line
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\x1b", "\\x1b")
    )
    py_lines.append(f'    "{escaped}\\n"')
body = "\n".join(py_lines)
out.write_text(
    '"""ANSI-art rendering of the Coral Bricks logomark.\n\n'
    "Generated offline from `Coralbricks/webapp/public/logo-icon.svg`. Do not\n"
    "hand-edit — rerun `scripts/regen_logo.sh` and paste the new output here.\n\n"
    "Rendered in truecolor so the coral red (#d44027) round-trips exactly\n"
    "instead of being quantized to the nearest 256-palette slot.\n"
    '"""\n\n'
    "from __future__ import annotations\n\n"
    "# Wordmark fill (`#d64027` from logo-with-text.svg). Matches the coral\n"
    "# that the icon already renders in, so the two halves read as one piece.\n"
    "WORDMARK_RGB = (214, 64, 39)\n\n"
    "# chafa --animate off --symbols block --size 24x10 --font-ratio=0.42 --colors full --format symbols\n"
    "LOGO_ICON_ANSI = (\n"
    f"{body}\n"
    ")\n"
)
PY

echo "wrote $out"
