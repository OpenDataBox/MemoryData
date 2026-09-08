# LoCoMo server queue handoff

## Active recovery on RTX 5090 (2026-09-07)

- Replacement instance `50118771` is reachable at `ssh4.vast.ai:38771` and
  provides one RTX 5090 (32 GB), CUDA toolkit 12.8, driver 580.159.03 and a
  50 GB container disk. The measured Vast total rate is
  `0.3959259259259259 USD/hour` before usage-priced bandwidth.
- Old instance `50064241` remains intact at `ssh2.vast.ai:24241`. Its
  container, SSH and Jupyter work, but both `nvidia-smi` and the live Vast
  capability manifest report no usable GPU after the host-side CUDA failure.
- A source-side zstd archive copied `HiGMem`, `MemoryData`, checkpoints,
  predictions, usage journals and queue artifacts to the replacement. SHA256
  matched on old server, local relay and new server:
  `3298989dfa3c7a61a824771326f75f15e34872e02876ef883e459a074f4c3a25`.
- Replacement verification found exactly 1417 prediction rows, 38632 usage
  rows and all ten checkpoint pickle files. The active recovery runner SHA256
  is `e832247c82a4e9e9dc7da9eb98e1918da7c6ade00af6b59f67cc8c0e72378902`.
- Qwen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, MiniLM revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, and LLMLingua revision
  `5f0c82792b7ea14c6484e015b6a072009496b7f2` were downloaded afresh on the
  replacement. vLLM 0.28.0 / Torch CUDA 13 runtime installation is in progress.
- The user explicitly requested that old instance `50064241` be destroyed
  only after replacement data verification, live inference verification and
  successful workload resumption. Do not destroy it before all three gates.

## Current authorization and shutdown condition

The user wants HiGMem followed by A-MEM, Mem0, LangMem, SimpleMem,
LightMem and E-Mem, then clarified that this Windows PC should shut down
**as soon as the independent server queue is registered and confirmed**.
Do not wait for all experiment results before shutting down the PC.
Do not stop or destroy the Vast instance. Do not force-close unsaved apps.
The user also authorized skipping any failed baseline and continuing with
the remaining methods; failed methods must not be reported as completed.

The server is `ssh2.vast.ai:24241`, instance `50064241`.
SSH identity file: `C:\Users\tgc04\.ssh\vast_codex_ed25519`.
Read `/etc/vast-agents-guide.md` before operating it.

## Source deployment approved and completed

After the source archive SCP was rejected twice, the user explicitly approved
the source/config transfer to `ssh2.vast.ai:24241`. The approved v2 archive
was uploaded and extracted into `/workspace/MemoryData`.

Deployed archive:
`D:\MemoryData\comparison_staging\source-20260907-queue-v2.tar.gz`
(443 source/config files, 703212 bytes, SHA256
`099321d4cc2359e1a04310bc6c98758b64eeddac6fff51291d77409b323096c5`).
It includes the reviewed venv-symlink, recovered-connection accounting,
LightMem source-alignment and executor error-propagation fixes. The older
v1 archive is stale and must not be deployed.

The separate earlier approval blocker for downloading remote checkpoints,
conversation-derived results and usage logs to this PC remains in force.
Source deployment approval does not authorize result export.

## Verified server state / prepared environment

### Recovery observed on 2026-09-07 at 01:17 UTC

- The overnight run actually exited at 19:47 UTC with 1417/1540 QA and
  5767/5882 checkpointed turns. `conv-44` stopped at turn 560/675 after
  three construction requests each timed out at 600 seconds. Its 123 QA
  were not run. The partial official F1 was 0.3979963461551367, NOT final.
- The queue exited at 19:48 UTC on its HiGMem completeness gate; none of
  the six baselines had started. A stale `waiting_for_higmem` JSON was not
  proof of a live process. Supervisor terminal status and traceback proved it.
- The original vLLM PID 3004 remained healthy with zero running/waiting
  requests. Local and deployed HiGMem runner SHA256 matched. The runner
  restores checkpoints and skips saved QA while appending usage; a plain
  supervised restart preserves experiment settings and prior 1417 answers.
- At 01:20 UTC, resumed HiGMem as PID 12749 and queue as PID 12754. The old
  partial report and queue state were preserved on the server as
  `report.before-resume-20260907T0120Z.json` and
  `queue_status.before-resume-20260907T0120Z.json` respectively. No result
  download, model restart or precision/generation change was performed.
- Fresh SSH at 01:21 UTC confirmed both recovery processes RUNNING and
  actual new model usage (38605 cumulative attempts, four failed attempts,
  zero observed truncations). Completion remains unproven.
- HiGMem `invocation_wall_seconds` is per invocation, not cumulative across
  resume. Preserve the prior report for accounting. Four failed attempts
  lack upstream usage and must remain unknown, never zero.
- The previous normal local PC shutdown was already requested. Do not
  repeatedly shut down a PC that has subsequently been turned back on.
  The old shutdown heartbeat remains PAUSED.
- At 01:26 UTC, HiGMem PID 12749 and queue PID 12754 remained RUNNING.
  Cumulative attempts increased to 38629 from 38595, with no new failed
  attempts or observed truncations. QA count remained 1417 while the
  incomplete conversation's memory was being constructed.
- A fresh no-model-API initializer ran for each of all six clients under
  `/workspace/locomo-preflight`. No method-specific import/init failure was
  observed; LightMem loaded the pinned CPU compressor. All six diagnostic
  commands nevertheless exited 1 because `preflight_locomo_clients.py:61`
  incorrectly checks the wrapper as a `PreTrainedTokenizerBase`. A separate
  deployed shared-loader probe confirmed `HuggingFaceTokenizerAdapter`
  wraps a genuine `Qwen2TokenizerFast` and the underlying instance passes
  that invariant. This is a diagnostic false positive, not a proven tokenizer
  fallback. Do not exclude these baselines based on that exit code. The
  preflight script is not called by the live queue. Full inference remains
  unverified. No source/model/service changes were made for this check.

### GPU terminal failure observed on 2026-09-07 at 01:47 UTC

- The resumed run repeated the same `conv-44` `_decide_event_affiliation`
  request. Attempts 0 and 1 timed out after 600 seconds. Attempt 2 ended in
  an HTTP 500 when vLLM's EngineCore failed with `CUDA error: unspecified
  launch failure`; the scheduler dump showed 5012 generated tokens for the
  pathological structured response. HiGMem and the queue then exited.
- The host-injected GPU is currently unavailable inside the container:
  `vast-capabilities` reports `No GPU detected`, and `nvidia-smi` reports
  `No devices were found` / `Unable to determine the device handle`.
  Do not restart the model process in this state.
- Current preserved result is still 1417/1540 QA and 5767/5882 checkpointed
  turns. Failed attempts increased from four to seven; their token use is
  unknown, not zero. The six baselines have still not begun.
- The narrow recovery candidate is to bound only event-affiliation JSON:
  disallow extra properties, bound the brief reasoning and affiliation list
  strings/count, and add a generous 4096-token safety ceiling for this call.
  Previous valid responses were 513-961 characters, so these constraints
  preserve intended semantics while stopping runaway output. Any protocol
  change must be recorded with the original manifest preserved.
- That recovery patch is now prepared locally in
  `D:\MemoryData\HiGMem\run_vast.py`: only `_decide_event_affiliation` gets
  bounded JSON fields plus a 4096-token safety ceiling. Other construction
  and QA calls are unchanged. SHA256 is
  `2b5497f2239c5f8773f2dda5966856e6865473fb89b47a7884d5537af91093e3`.
  `test_vast_runner.py` passed all 6 tests under the existing Python 3.12
  comparison environment, followed by a successful `py_compile`. It has
  not been uploaded or applied to the server pending explicit approval.
- Vast CLI documents `reboot instance` as stop/start without loss of GPU
  priority, and the instance guide says stop/start preserves the container
  filesystem. A reboot is nevertheless a material external action and was
  not executed. User approval is required because the earlier instruction
  was not to stop or destroy the Vast instance. After approval, reboot,
  verify GPU health, start vLLM and wait for health, then resume HiGMem and
  the queue in that order. Never destroy or recycle the instance.

### Historical deployment snapshot (superseded by recovery above)

- HiGMem is independently supervised as `higmem-eval` (PID 3597) and
  `higmem-vllm` (PID 3004). Do not restart while healthy.
- Latest service observation: HiGMem remained RUNNING at 4:32 runtime.
  The preceding 4:26 aggregate snapshot had 389/1540 QA, 5137 checkpointed
  turns, 21991 requests, 21157872 reported prompt tokens and 4052210
  completion tokens; 1 failed request and 0 truncated responses.
- One 4ms `Connection error` for `conv-49`, QA index 61 was followed by
  metered successful retries and a saved nonempty answer. Its failed-attempt
  tokens remain unknown; do not replace them with zero or claim exact totals.
- `/workspace/comparison-venv` was created separately. Its `.pth` shares the
  existing Torch/CUDA packages read-only; 47 non-GPU client packages were
  installed locally to that new environment. The live HiGMem environment
  was not modified. Import probe succeeded with Torch 2.13.0+cu130,
  Transformers 4.57.6, HTTPX 0.28.1, LangChain, LanceDB, Qdrant and LLMLingua.
- Pinned LLMLingua auxiliary CPU model is downloaded. Revision
  `5f0c82792b7ea14c6484e015b6a072009496b7f2`; weight SHA256
  `22b9ecde52fec5c97e8c54a293be768727df95a81c6c8dccb03f262a50c58324` verified.
- Disk remaining after environment/model preparation: approximately 4.2 GB.
- Actual read-only Vast price query: **0.3422222222222222 USD/hour total**,
  base GPU price 0.3333333333333333 USD/hour; disk component unavailable.
- The six-method source tree and plan are deployed. Plan preparation and
  `--check-only` passed for all six methods, each with the full 1540 QA.
- Supervisor registered both new programs. `locomo-comparison-queue` is
  RUNNING as PID 8357 since approximately 2026-09-06 18:40 UTC.
  `comparison-vllm` remains STOPPED intentionally while the original
  `higmem-eval` and `higmem-vllm` services remain RUNNING.
- Fresh SSH independently confirmed queue PID 8357 RUNNING (32 seconds
  uptime), parent PID 628 (Supervisor), and `queue_status.json` state
  `waiting_for_higmem`. HiGMem PID 3597 remained RUNNING at 4:32:42 runtime
  and its original vLLM PID 3004 remained RUNNING.
- Local heartbeat `automation` was changed to PAUSED and verified. The
  reservation condition for the user's normal local PC shutdown is met.

## Deployment and shutdown handoff

1. Completed: approved v2 source deployment, manifest validation, plan
   preparation and check-only validation. Plan is at
   `/workspace/MemoryData/comparison_config/plan.json`; output root is
   `/workspace/locomo-comparison-qwen35-fp16`; run ID is
   `locomo-qwen35-fp16-20260907`, hourly rate is 0.3422222222222222 USD.
2. Completed: both Supervisor programs registered, only the queue started.
   The queue waits for HiGMem to exit and pass output/data gates before
   replacing its vLLM service with `comparison-vllm`. Individual baseline
   failures are recorded in `failures.json` and the next method is attempted.
3. Completed: fresh SSH confirmed the supervised queue is RUNNING,
   waiting for HiGMem, while the current HiGMem remains healthy.
4. Completed: local shutdown heartbeat paused. Leave server paths/state for
   the user, then perform a normal Windows shutdown without `/f`.
   Registration is not a claim that all six experiments have succeeded.

## Fixed comparison and accounting

- Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
  FP16 weights and nonquantized KV cache, 32768 context, vLLM prefix caching.
- Explicit opt-in proxy policy: temperature 0, thinking disabled, request output
  caps removed, `--generation-config vllm`. Messages, tools and schemas are
  preserved. Truncated output is rejected only after recording actual usage.
- Same MiniLM CPU encoder as HiGMem: revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions and native
  256-token encoder truncation accounted separately.
- Original public LoCoMo category 1–4: 1540 QA, 5882 complete turns.
  Per-turn ingestion preserves original speaker, date, caption and source ID.
- A-MEM retrieval 10; SimpleMem original 40-turn window / overlap 2 with
  planning/reflection; LightMem full compression/topic/extraction/offline
  update pipeline, retrieval 60; E-Mem explicitly text mode.
- Actual API usage is recorded per attempt, including internal operations.
  LightMem local auxiliary model forwards/tokens/time are separate. Official
  category-aware F1 checks exact IDs and original metadata. GPU energy is
  sampled/integrated, not a hardware-counter reading; missing values stay null.
- A recovered HiGMem transport error may permit subsequent methods after all
  original questions complete, but its accounting sidecar explicitly marks
  unknown failed-attempt tokens. This does not rewrite HiGMem's original logs.

Final combined local verification before v2 packaging: **148 passed,
1 skipped, 2 warnings** in 79.28 seconds. The skipped test requires creation
of a real venv symlink, unavailable with the Windows test permissions.
The final Python review reported no remaining blockers.
