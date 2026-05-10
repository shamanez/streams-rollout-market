# Evidence map

The goal of this file is to keep the project grounded in the current pain points around agentic RL rollout scaling, training-inference mismatch, staleness, quantized rollout, MoE routing, spot compute, and agent-built development.

1. **Accelerating RL Post-Training with Speculative Decoding in NeMo RL** (2026-04; rollout bottleneck)
   - Why it matters: Reports rollout generation as the dominant RL post-training cost; 65-72% of step time in 8B workloads, with 1.5x-1.8x rollout speedups and up to 1.4x end-to-end step speedup.
   - URL: https://research.nvidia.com/labs/nemotron/rl-speculative-decoding/
2. **ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents** (2026-03; rollout as service)
   - Why it matters: Directly validates that rollout orchestration should be decoupled from the trainer and exposed as an API service for multi-turn agent RL.
   - URL: https://www.microsoft.com/en-us/research/publication/prorl-agent-rollout-as-a-service-for-rl-training-of-multi-turn-llm-agents/
3. **SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling** (2026-03; rollout bottleneck)
   - Why it matters: Reports rollout phase can account for up to 70% of training time for long trajectories and motivates scheduling plus controlled off-policy replay.
   - URL: https://www.microsoft.com/en-us/research/publication/sortedrl-accelerating-rl-training-for-llms-through-online-length-aware-scheduling/
4. **Heddle: Agentic RL Training Needs Trajectory-Centric Scheduling** (2026-03; rollout bottleneck)
   - Why it matters: Post-Jan-2026 system direction: long-tailed tool calls and trajectory placement become first-class scheduling problems. Use as a search/triage placeholder if exact arXiv changes.
   - URL: https://arxiv.org/abs/2603.28101
5. **Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding** (2026-04; speculative rollout)
   - Why it matters: Formalizes lossless rollout acceleration where verifier policy semantics are preserved, distinguishing pure throughput from off-policy rollout changes.
   - URL: https://arxiv.org/abs/2604.26779
6. **VeRL Rollout Importance Sampling documentation** (2025-10; training-inference mismatch)
   - Why it matters: Operationalizes rollout-vs-training mismatch metrics and rollout IS; useful blueprint for OPBC telemetry.
   - URL: https://verl.readthedocs.io/en/latest/advance/rollout_is.html
7. **When Speed Kills Stability: RL Collapse from Inference-Training Mismatch** (2025-10; training-inference mismatch)
   - Why it matters: Core conceptual reference for treating inference engine mismatch as a behavior-policy mismatch.
   - URL: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Inference-Training-Mismatch-271211a558b7808d8b12d403fd15edda
8. **Defeating the Training-Inference Mismatch via FP16** (2025-10; training-inference mismatch)
   - Why it matters: Argues BF16 precision itself can create mismatch and that FP16 can improve RL stability; motivates precision as marketplace metadata.
   - URL: https://huggingface.co/papers/2510.26788
9. **Precision-RL repository** (2025-11; training-inference mismatch)
   - Why it matters: Tracks implementation support for FP16 RL and experiments across algorithms/model families including MoE.
   - URL: https://github.com/sail-sg/Precision-RL
10. **No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL with vLLM and TorchTitan** (2025-11; training-inference mismatch)
   - Why it matters: Shows the ecosystem is moving from correction-only toward bitwise/numerical consistency as a systems requirement.
   - URL: https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html
11. **No More Retokenization Drift: Returning Token IDs via OpenAI Compatible API Matters in Agent RL** (2025-10; token integrity)
   - Why it matters: Supports the hard contract that token IDs must be returned across API boundaries to avoid retokenization drift.
   - URL: https://vllm.ai/blog/agent-lightning
12. **Swift Training-Inference-Mismatch documentation** (2026; training-inference mismatch)
   - Why it matters: Explains GRPO assumptions and IS correction for rollout/training policy mismatch in practitioner documentation.
   - URL: https://swift.readthedocs.io/en/v4.0/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.html
13. **Router Replay R3: Why It Failed and How We Fixed It** (2026-01; MoE mismatch)
   - Why it matters: Explains MoE-specific training-inference mismatch, routing replay, and failure modes when correction discards too many samples.
   - URL: https://macaron.im/mindlab/research/router-replay-r3-why-it-failed-and-how-we-fixed-it
14. **Rollout Routing Replay / R3 paper** (2025-10; MoE mismatch)
   - Why it matters: Key MoE mismatch primitive: record rollout router choices and replay them during training.
   - URL: https://huggingface.co/papers/2510.11370
15. **Qwen1.5-MoE-A2.7B model card** (2024-03; MoE model candidate)
   - Why it matters: Practical small MoE candidate: 14.3B total, 2.7B activated, with vLLM support according to the model card.
   - URL: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
16. **Qwen1.5-MoE-A2.7B-Chat model card** (2024-03; MoE model candidate)
   - Why it matters: Chat variant; model card states 14.3B total/2.7B activated and faster inference than Qwen1.5-7B.
   - URL: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat
17. **Qwen1.5-MoE blog** (2024-03; MoE model candidate)
   - Why it matters: Explains why the model is a good small MoE demo: 2.7B active parameters, 75% training cost reduction, 1.74x inference speed vs Qwen1.5-7B.
   - URL: https://qwenlm.github.io/blog/qwen-moe/
18. **QuRL: Efficient Reinforcement Learning with Quantized Rollout** (2026-02; quantized rollouts)
   - Why it matters: ICLR 2026 quantized rollout work; rollout can be up to 70% of training time and quantized actors require adaptive correction.
   - URL: https://openreview.net/forum?id=eG0bpCwdKn
19. **QaRL: Rollout-Aligned Quantization-Aware RL** (2026-04; quantized rollouts)
   - Why it matters: Highlights quantized rollout instability via training-inference gap and long-form error tokens, especially relevant to MoE.
   - URL: https://arxiv.org/abs/2604.07853
20. **NeMo RL Quantization-Aware RL guide** (2026; quantized rollouts)
   - Why it matters: Practical evidence that quantizer state must be transferred/refit through the RL loop.
   - URL: https://docs.nvidia.com/nemo/rl/nightly/guides/quantization-aware-rl.html
21. **QeRL: Quantization-enhanced Reinforcement Learning for LLMs** (2025-10; quantized rollouts)
   - Why it matters: Shows quantization is not just a speed knob; it changes exploration and policy entropy, so marketplace metadata must expose it.
   - URL: https://research.nvidia.com/labs/eai/publication/qerl/
22. **VeRL FP8 rollout documentation** (2026-03; quantized rollouts)
   - Why it matters: Implementation details for BF16 training with FP8 rollout and end-to-end FP8, motivating precision-aware contracts.
   - URL: https://verl.readthedocs.io/en/latest/perf/fp8.html
23. **Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs** (2026-02; async staleness)
   - Why it matters: Connects stale rollouts to heavy-tailed importance weights and ESS collapse; validates OPBC metrics.
   - URL: https://huggingface.co/papers/2602.17616
24. **Stable Asynchrony emergentmind summary** (2026-02; async staleness)
   - Why it matters: Summarizes VCPO and the use of ESS to control high-asynchrony policy-gradient variance.
   - URL: https://www.emergentmind.com/papers/2602.17616
25. **Prosperity before Collapse: M2PO** (2025-10; async staleness)
   - Why it matters: Shows stale rollouts can be useful if second-moment trust is controlled; motivates keeping replay but scoring it.
   - URL: https://infini-ai-lab.github.io/M2PO/
26. **BAPO: Adaptive Clipping for Diverse Off-Policy Scenarios** (2026; async staleness)
   - Why it matters: Evidence that off-policy clipping should adapt to scenario rather than use one static threshold.
   - URL: https://openreview.net/forum?id=jIeJJqG7dz
27. **VESPO: Variance-Enhanced Stable Policy Optimization** (2026; async staleness)
   - Why it matters: Identifies policy staleness and training-inference mismatch as behavior/current policy divergence; source to verify during implementation.
   - URL: https://arxiv.org/abs/2602.10693
28. **AReaL: Asynchronous Reinforcement Learning for LLMs** (2025-05; async RL framework)
   - Why it matters: Decoupled generation/training with bounded policy lag; central baseline for async rollout systems.
   - URL: https://arxiv.org/abs/2505.24298
29. **ROLL Flash: Efficient Async RL** (2025-10; async RL framework)
   - Why it matters: Async RL design with ratio scheduling and agentic task speedups; useful baseline for marketplace comparisons.
   - URL: https://arxiv.org/abs/2510.11345
30. **SkyRL-Agent** (2025-11; async RL framework)
   - Why it matters: Agent training/evaluation framework with long-horizon multi-turn RL concerns and backend interoperability.
   - URL: https://arxiv.org/abs/2511.16108
31. **VeRL fully_async documentation** (2025; async RL framework)
   - Why it matters: Practical async RL knobs; important for trainer-client integration examples.
   - URL: https://verl.readthedocs.io/en/latest/examples/config.html
32. **Agent Lightning** (2025-08; async RL framework)
   - Why it matters: Contract-oriented transition/span store and API style relevant to streams contracts.
   - URL: https://arxiv.org/abs/2508.03680
33. **INTELLECT-2 Release** (2025-05; decentralized RL)
   - Why it matters: Shows globally distributed asynchronous RL with TOPLOC rollout verification and SHARDCAST weight broadcast.
   - URL: https://www.primeintellect.ai/blog/intellect-2-release
34. **Prime-RL async documentation** (2026; decentralized RL)
   - Why it matters: Practical async/decentralized RL framework with behavior-policy/current-policy corrections.
   - URL: https://github.com/PrimeIntellect-ai/prime-rl
35. **INTELLECT-3 release** (2026; decentralized RL)
   - Why it matters: Demonstrates 100B+ MoE distributed RL direction, reinforcing the need for async rollout protocols.
   - URL: https://www.primeintellect.ai/blog/intellect-3-release
36. **OpenDiLoCo** (2024-07; decentralized training)
   - Why it matters: Useful prior for decentralized compute utilization and intermittent synchronization, though not directly rollout-market specific.
   - URL: https://arxiv.org/abs/2407.07852
37. **Streaming DiLoCo** (2025-01; decentralized training)
   - Why it matters: Bandwidth-reduction primitive relevant to weight-update propagation under churn.
   - URL: https://arxiv.org/abs/2501.18512
38. **TOPLOC** (2025; verification)
   - Why it matters: Approximate verification of rollout/inference claims across devices; conceptually relevant to policy attestation.
   - URL: https://arxiv.org/abs/2501.16007
39. **PULSE sparse patch broadcast** (2026-02; weight sync)
   - Why it matters: Suggests RL weight deltas may be sparse enough for frequent patch transport, a key rollout-market primitive.
   - URL: https://arxiv.org/abs/2602.03839
40. **vLLM weight-sync RFC / layerwise reloading threads** (2026; weight sync)
   - Why it matters: Operational signal that production weight reload paths are non-trivial and need contracts, ACKs, and audits.
   - URL: https://github.com/vllm-project/vllm/issues
41. **Vast.ai instance types documentation** (2026; spot GPU marketplace)
   - Why it matters: Defines on-demand, reserved, interruptible; interruptible instances can be paused if outbid or on-demand requested.
   - URL: https://docs.vast.ai/documentation/instances/choosing/instance-types
42. **Vast.ai rental types FAQ** (2026; spot GPU marketplace)
   - Why it matters: Direct-bidding, 50-80% cost savings, and interruptions create a realistic churn model for rollout workers.
   - URL: https://docs.vast.ai/guides/reference/faq/rental-types
43. **RunPod savings / spot instance documentation** (2026; spot GPU marketplace)
   - Why it matters: Spot instances may terminate with a short SIGTERM window, requiring checkpointing and trajectory commit protocols.
   - URL: https://docs.runpod.io/pods/savings-plans
44. **RunPod pricing guide** (2026; spot GPU marketplace)
   - Why it matters: Pricing input for cost-per-accepted-rollout-token benchmarks.
   - URL: https://www.runpod.io/pricing
45. **Fireworks completion API logprobs** (2026; endpoint probe)
   - Why it matters: Free/cheap endpoint probe candidate because completions expose sampled-token logprobs and top_logprobs.
   - URL: https://docs.fireworks.ai/api-reference/post-completions
46. **Fireworks chat completion API logprobs** (2026; endpoint probe)
   - Why it matters: Useful for endpoint identity gap demo, with warnings about provider-specific logprob semantics.
   - URL: https://docs.fireworks.ai/api-reference/post-chatcompletions
47. **Together AI OpenAI-compatible API docs** (2026; endpoint probe)
   - Why it matters: Endpoint probe candidate for comparing identical model labels across providers.
   - URL: https://docs.together.ai/docs/openai-api-compatibility
48. **OpenRouter API docs and logprobs parameters** (2026; endpoint probe)
   - Why it matters: Router/proxy case study: great for demonstrating that model label, provider, and logprob semantics must be explicit.
   - URL: https://openrouter.ai/docs/api-reference/parameters
49. **vLLM documentation** (2026; inference engine)
   - Why it matters: Primary local rollout serving engine for controlled dense/MoE experiments.
   - URL: https://docs.vllm.ai/
50. **SGLang documentation** (2026; inference engine)
   - Why it matters: Second local rollout serving engine for mismatch decomposition and MoE routing experiments.
   - URL: https://docs.sglang.ai/
51. **ROCm vLLM documentation / AMD blogs** (2026; hardware heterogeneity)
   - Why it matters: Supports CUDA/ROCm heterogeneity as a practical rollout-market issue.
   - URL: https://rocm.blogs.amd.com/artificial-intelligence/vllm-optimize/README.html
52. **Liger Kernel ROCm support** (2026; hardware heterogeneity)
   - Why it matters: Kernel-level cross-platform support is improving, but end-to-end RL-loop numerical fidelity remains open.
   - URL: https://github.com/linkedin/Liger-Kernel
53. **Helix heterogeneous inference scheduling** (2025; scheduler)
   - Why it matters: Layer placement and routing over heterogeneous GPU clusters; useful background for worker capability routing.
   - URL: https://arxiv.org/abs/2406.01566
54. **HexGen-2** (2025; scheduler)
   - Why it matters: Prefill/decode disaggregation and heterogeneous scheduling inform long-context rollout routing.
   - URL: https://arxiv.org/abs/2502.07903
55. **Llumnix** (2025; scheduler)
   - Why it matters: Live KV migration concepts relevant to worker churn, though cross-vendor KV portability remains hard.
   - URL: https://github.com/AlibabaPAI/llumnix
56. **NeMo Gym** (2026; agent RL environment)
   - Why it matters: External environment harness integration target; the rollout market should plug into env providers rather than own training.
   - URL: https://github.com/NVIDIA/NeMo-Gym
57. **OpenHands / SWE-agent style environments** (2025; agent RL environment)
   - Why it matters: Representative long-horizon coding agent environment with tool calls and sandbox behavior.
   - URL: https://github.com/All-Hands-AI/OpenHands
58. **SWE-bench Verified** (2026; benchmarks)
   - Why it matters: Target task family where rollouts can be long, tool-heavy, and variable-latency.
   - URL: https://www.swebench.com/
59. **Ionides truncated importance sampling** (2008; policy gradient theory)
   - Why it matters: Foundational theory for heavy-tailed importance weights and clipping/truncation tradeoffs.
   - URL: https://doi.org/10.1198/106186008X320456
60. **IMPALA / V-trace** (2018; policy gradient theory)
   - Why it matters: Actor-learner off-policy correction baseline; useful but not enough for LLM long-horizon token-level ratios.
   - URL: https://arxiv.org/abs/1802.01561
61. **Decoupled PPO** (2022; policy gradient theory)
   - Why it matters: Formalizes decoupling behavior and proximal policies; relevant to async RL trainers consuming stale groups.
   - URL: https://arxiv.org/abs/2209.01932
62. **GSPO** (2025; policy gradient objective)
   - Why it matters: Sequence-level ratio alternative with different variance/credit assignment tradeoffs.
   - URL: https://arxiv.org/abs/2507.18071
63. **TIC-GRPO** (2026; policy gradient objective)
   - Why it matters: GRPO convergence/correction analysis; relevant to one-policy-snapshot-per-group contracts.
   - URL: https://arxiv.org/abs/2508.02833
64. **SORL / turn-level IS for tool use** (2025; tool RL)
   - Why it matters: Supports per-turn or segmented IS because tool-output tokens should be masked and long trajectories explode token-level ratios.
   - URL: https://arxiv.org/abs/2511.20718
65. **Turn-PPO / turn-level MDP proposals** (2026; tool RL)
   - Why it matters: Motivates a future OPBC feature: turn-level budget rather than only token-level or sequence-level metrics.
   - URL: https://arxiv.org/abs/2512.17008
66. **OpenAI Codex cloud documentation** (2026; agent coding)
   - Why it matters: Confirms Codex can read, modify, and run code in sandboxed cloud containers and work on parallel tasks.
   - URL: https://platform.openai.com/docs/codex
67. **Introducing Codex** (2025-05; agent coding)
   - Why it matters: Explains AGENTS.md guidance, cloud sandbox task execution, test logs, and PR-style workflows.
   - URL: https://openai.com/index/introducing-codex/
68. **How OpenAI uses Codex** (2026; agent coding)
   - Why it matters: Recommends issue-like prompts, startup scripts, reliable environments, and AGENTS.md persistent context.
   - URL: https://openai.com/business/guides-and-resources/how-openai-uses-codex/
69. **Claude Code Agent SDK / headless mode** (2026; agent coding)
   - Why it matters: Supports scripted long-running implementation with claude -p, tool permissions, and structured output.
   - URL: https://code.claude.com/docs/en/headless
70. **Claude Code headless docs** (2026; agent coding)
   - Why it matters: Confirms non-interactive CLI execution, JSON output, resume, and allowed tools.
   - URL: https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless
