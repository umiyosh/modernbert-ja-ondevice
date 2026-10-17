#!/bin/sh
# Wait for a nightly job to end, write its report, and commit the code, reports and progress log so
# each night leaves one commit. Pushing is left until someone has read the report.
#
# The report command must print its commit subject on a last line starting with "COMMIT: ".
#
# usage: asr/nightly.sh <pid of the job> <report command...>
set -u
pid=$1
shift
repo=$(cd "$(dirname "$0")/.." && pwd)
while kill -0 "$pid" 2>/dev/null; do sleep 60; done

out=$("$@") || { echo "report failed; nothing committed" >&2; exit 1; }
msg=$(printf '%s\n' "$out" | grep '^COMMIT: ' | tail -1 | cut -c9-)
[ -n "$msg" ] || { echo "report printed no COMMIT line; nothing committed" >&2; exit 1; }

git -C "$repo" add -- asr/*.py asr/*.sh asr/README.md asr/PROGRESS.md asr/reports
git -C "$repo" commit -q -m "$msg" && git -C "$repo" log -1 --oneline
