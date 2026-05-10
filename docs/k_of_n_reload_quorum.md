# k-of-n reload quorum

## Problem

When the trainer publishes a new `PolicyManifest`, every worker in the pool
needs to reload weights before it can produce rollouts under the new
version. With a fully synchronous reload barrier, *one* slow worker (a
spot interruption, a slow disk, a paused container) blocks the entire
trainer step. With a fully asynchronous reload, the dispatcher cannot
tell the trainer "you can start consuming version vN" without risking a
mixed-version group.

The decentralized rollout marketplace has the same problem with extra
constraints:

- Workers may be heterogeneous (different engines, precisions, hardware)
  and reload at different speeds.
- Spot workers can disappear in the middle of a reload.
- A trainer client can ask for a specific `policy_version` at any time;
  it must never silently get rollouts from a different version.

## Design

We split *pool-level visibility* from *per-group pinning*.

### Pool-level visibility (quorum)

`PoolReloadTracker` records `ReloadAck(worker_id, policy_version,
checkpoint_digest, acked_at)` whenever a worker reports it has loaded a
new manifest. The tracker exposes:

- `record_ack(ack)` — idempotent on `worker_id`; a worker can only be in
  one quorum at a time.
- `has_quorum(version, k=, n=)` — true when `ack_count(version) >= k`.
- `pool_active_version(k=, n=, candidates=)` — the latest version that has
  reached quorum.
- `is_worker_aligned(worker_id, policy_version)` — the canonical
  per-worker check.

The pool advances its declared "active" version as soon as the **k-th**
of n workers acks. The trainer can read `pool_active_version` to start
producing jobs against vN without waiting for stragglers.

### Per-group pinning (strict)

Per-group pinning is unchanged from earlier phases:

- Every `RolloutJob` carries `policy_version`, `checkpoint_digest`,
  `tokenizer_hash`.
- Every `RolloutLease` echoes `policy_version`.
- Every `SampleGroup` echoes `policy_version`, `checkpoint_digest`,
  `tokenizer_hash`.
- `validators.validate_group_against_lease` rejects with
  `RejectionReason.POLICY_VERSION_MISMATCH` if the group's
  `policy_version` differs from the lease's, and with
  `RejectionReason.MIXED_POLICY_VERSION` if it differs from the
  manifest the registry currently advertises.

Quorum **never** loosens these checks. A worker that has not acked vN
will not be issued a vN lease, because the dispatcher's per-worker gate
is `tracker.is_worker_aligned(worker_id, policy_version)`. A worker that
has acked vN but tries to submit a vN-1 group will be rejected by the
existing validator chain.

### Quorum violation

If two workers ack the same `policy_version` with different
`checkpoint_digest` values, the tracker raises `QuorumViolation`. This
is a contract bug — the trainer should not be publishing the same
version with different weights — and we surface it loudly rather than
silently letting one digest win.

## Invariants

1. **Strict per-group pinning.** No SampleGroup is ever accepted with a
   `policy_version` that differs from its lease.
2. **No phantom advances.** `pool_active_version(k, n)` returns `None`
   if no version has k acks.
3. **One worker, one ack.** A worker that re-acks for a new version is
   removed from the quorum of any older version it had acked.
4. **Digest agreement.** Two workers cannot ack the same version with
   different checkpoint digests without raising `QuorumViolation`.
5. **Old leases stay valid.** A worker that holds a lease pinned to vN-1
   is unaffected when the pool advances to vN; the lease will SUBMIT or
   EXPIRE under vN-1 semantics. Mixed-version groups are still rejected.

## Operational notes

- The trainer chooses `k` and `n`. A common choice is `k = ceil(0.5 * n)
  + 1` (strict majority). For small pools (n=3, n=4), `k = n - 1` is
  often more useful: it tolerates exactly one straggler.
- The tracker is in-process and thread-safe. A multi-process deployment
  swaps it for a small KV-backed implementation with the same surface.
- Dispatcher integration: gate every job dispatch on
  `is_worker_aligned`, *not* on `pool_active_version`. The latter
  answers "what does the trainer think the pool is on?" — the former
  answers "is this specific worker safe to give this specific job to?"
