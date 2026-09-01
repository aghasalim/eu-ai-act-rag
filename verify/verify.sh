#!/usr/bin/env bash
# Recompute the published numbers in every language here and require agreement.
#
# Every figure in README.md and RESULTS.md comes out of eval/run_eval.py and
# eval/report.py. So does every figure image. If the averaging or the metric
# maths in there were wrong, nothing downstream would notice, because everything
# downstream reads the same output and the tests check that the code runs rather
# than that it is right. These are independent implementations reading the same
# raw per-question records, and a mistake would have to be repeated identically
# in all of them to survive.
#
# Each is skipped with a message if its toolchain is missing, so this still runs
# something useful on a laptop with only part of the set installed. CI has all
# of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# rustup installs outside the default PATH on a workstation. Harmless when the
# toolchain is somewhere else, as it is on the CI runner.
PATH="$HOME/.cargo/bin:$PATH"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL cannot fail a run by itself, so its output is inspected here. It prints a
# FAIL row per disagreement and a count of the comparisons it made; no rows at
# all would mean it read nothing, which is not a pass.
check_sql () {
    local out
    out=$(sqlite3 -init verify/aggregates.sql :memory: "" 2>&1)
    if grep -q '^FAIL' <<<"$out"; then
        grep '^FAIL' <<<"$out" | head -20
        return 1
    fi
    local n
    n=$(sed -n 's/^CHECKED|//p' <<<"$out")
    if [ -z "$n" ] || [ "$n" -eq 0 ]; then
        echo "no comparisons were made:"; echo "$out" | head -20
        return 1
    fi
    printf '  %s aggregates recomputed from the per-question numbers, worst |d| %s\n' \
           "$n" "$(sed -n 's/^MAXDELTA|//p' <<<"$out")"
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/euactrag_kernel" verify/kernel.c -lm || return 1
    "${TMPDIR:-/tmp}/euactrag_kernel" "$root"
}

check_go   () { ( cd verify/gocheck   && go run . -root "$root" ); }
check_rust () { ( cd verify/bootstrap && cargo run --release --quiet -- "$root" ); }

run "SQL, retrieval aggregates"    sqlite3 check_sql
run "C, retrieval metric kernel"   cc      check_c
run "Go, file structure"           go      check_go
run "Ruby, failure taxonomy"       ruby    ruby verify/failures.rb "$root"
run "JavaScript, published tables" node    node verify/published.js "$root"
run "R, statistical inference"     Rscript Rscript verify/inference.R "$root"
run "Rust, bootstrap intervals"    cargo   check_rust

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
