# A62 protected publication protocol

This is the implementation protocol, not an integration receipt. Actual main
apply calls are zero at this document's creation. Scientific completion and
protected publication are separate gates.

A62 source commits are made only in the dedicated A62 worktree. The original
main worktree remains attached to its existing branch. The original user index
and unrelated worktree bytes are never staged, reset, checked out, converted or
restored as a shortcut. All current user changes, including changes after the
initial snapshot, take precedence over old expected values.

The standalone `event_risk_git_integration.py` is a new A62 implementation. It
retains the reviewed A61 algorithm's quarantined dry run, exact new-file diff,
original index-entry preservation, expected-old prepared ref transaction,
index/HEAD locks, durable intent and original-index backup. The old helper is
unchanged and its A61 allowlist is not reused. The A62 helper allows only the
explicitly enumerated A62 plan/authorization files, new A62 result paths,
`event_risk` modules/scripts/tests and the A62 config. Every payload still needs
an explicit SHA256 and must be absent from both current HEAD and index.

The helper checks configured/default executable hooks and rejects a newly added
user gate rather than bypassing it. It also rejects Git environment overrides,
active sequencers/merges, split/sparse configuration/indexes, skip-worktree
entries and unmerged stages. Eligibility is checked again for apply and before
publication. Unknown or changed state fails closed. A preexisting foreign lock
is never removed. There is no broad lock cleanup or force update.

Dry run is the default. Actual apply needs current expected HEAD, full attached
branch, the dry run's original-index SHA256 and the same exact payload pins.
New payloads must be copied exclusively, verified against the sealed public
manifest, and rechecked as actual Git objects after publication. A commit adds
only those new paths and never includes user-staged theory or other user edits.

Index and ref publication are not crash-atomic. A partial result retains owned
markers, durable intent and backups for explicit reconciliation; it does not
automatically roll back or overwrite concurrent user state. A failed apply is
not evidence that nothing was published. The actual branch, index and journal
must be read before further action.

`docs/EXPERIMENT_INDEX.md` is outside the helper's commit allowlist. Its current
bytes may receive only the final A62 appendix, kept unstaged, after verifying the
current prefix again. The stale Index from the A62 scientific parent must never
replace the current main document.

The initial main preservation proof covers the raw current index plus a stated
selected content inventory and tracked metadata. Its 28 preexisting timed-out
content reads remain unresolved, not newly hashed. Final protection reports
must describe this scope accurately; the full private inventory and backup stay
local. No publication may claim all arbitrary untracked files were enumerated.

All helper checks use isolated synthetic Git repositories until final actual
closure. Passing those checks is not a real main integration or model result.
