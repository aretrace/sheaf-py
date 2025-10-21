#!/bin/sh
set -eu

# Usage:
#   ./convert.sh                # process ./*.jsonl in current dir
#   ./convert.sh path/to/dir    # process *.jsonl in that dir (non-recursive)
#   ./convert.sh path/to/file.jsonl  # process just that file

convert_file() {
  f=$1
  base=$(basename "$f")
  dirn=$(dirname "$f")
  out="$dirn/${base%.*}.json"
  tmp="$out.tmp.$$"

  if jq -s '.' "$f" > "$tmp"; then
    mv "$tmp" "$out"
    echo "Wrote $out"
  else
    echo "Error converting $f" >&2
    rm -f "$tmp"
    exit 1
  fi
}

# if no arguments, default to current directory
if [ "$#" -eq 0 ]; then
  set -- "."
fi

found=0

for path in "$@"; do
  if [ -d "$path" ]; then
    any=0
    # non-recursive: only files directly in the directory
    for f in "$path"/*.jsonl; do
      [ -e "$f" ] || continue
      any=1
      found=1
      convert_file "$f"
    done
    [ "$any" -eq 1 ] || echo "No .jsonl files found in $path."
  elif [ -f "$path" ]; then
    case "$path" in
      *.jsonl)
        found=1
        convert_file "$path"
        ;;
      *)
        echo "Skipping non-.jsonl file: $path"
        ;;
    esac
  else
    echo "Not found: $path" >&2
    exit 1
  fi
done

[ "$found" -eq 1 ] || { echo "No .jsonl files found."; exit 0; }
