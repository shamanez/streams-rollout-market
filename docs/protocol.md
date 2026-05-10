# Protocol notes

## Policy pinning

A group assignment must include the policy version before any sibling trajectory begins. The worker must submit the same policy version, checkpoint digest, tokenizer hash, sampling config hash, precision class, and engine contract for every trajectory in the group.

## Token contract

Services pass token IDs, not decoded strings. Tool-output tokens must be masked out of policy-gradient and importance-ratio accounting.

## Group lifecycle

pending -> accepted | correctable | quarantined | rejected

- accepted: safe to train directly.
- correctable: train only with configured correction.
- quarantined: keep for analysis, do not train.
- rejected: invalid contract or integrity failure.
