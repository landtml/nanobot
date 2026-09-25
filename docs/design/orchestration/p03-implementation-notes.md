# P03 scheduler implementation notes

`ScheduledProvider` is installed by `providers.factory.make_provider` when
called with `scheduled=True`, and by `schedule_provider` when
`AgentLoop.from_config` receives an injected provider. Provider snapshots and
`AgentLoop.from_config` use the scheduled form.
Direct `make_provider` calls keep their concrete provider return type by
default. The scheduler wraps the primary leaf and fallback leaves when the
fallback factory creates them.
`FallbackProvider` uses the typed `LaneSaturatedError` admission signal only for
locally full lanes. It skips those candidates without synthesizing an LLM error,
changing provider failure counters, or firing fallback retry callbacks. When
all candidates are locally full, it waits through the scheduler's fair,
cancellation-safe lane queues and carries the granted lease into the matching
candidate call. It does not release and reacquire the selected slot, which would
let another queued request take it first. This extends the phase touch list at
the fallback candidate seam because wrapping the entire fallback chain as one
lane would either queue before trying an available fallback or misreport local
saturation as a provider failure. The reservation is claimed only once; retry
attempts within the admitted candidate reacquire through the scheduler so they
honor concurrent occupancy and retry-after blocks.

`ProviderAdmissionError` is rethrown by the base provider's safe chat and stream
entry points. This is required on the runner's real `chat_with_retry` and
`chat_stream_with_retry` paths: treating local admission as an LLM error would
run provider retry/observer behavior and could poison the fallback breaker.
Scheduler feedback normalizes HTTP 429 responses, including `error_kind="http"`,
and escaped 429 exceptions. It applies AIMD decrease and honors response or
exception retry-after metadata.

The deterministic service-time simulator measures actual p95 admission waits
for root B. Ten existing roots make its baseline 0.11 s; adding A's 10-way
fan-out raises it to 0.13 s (18.2%). The equivalent FIFO schedule reaches
0.21 s, so the workload distinguishes root fairness from FIFO. A concurrent
429 storm test admits four requests initially, verifies later first attempts
run at a reduced lane cap, then checks that all eight requests succeed and the
cap recovers to four. Separate cancellation tests cover a lease granted as its
waiter is cancelled and an `acquire_any` winner granted as its caller is
cancelled. A raw-versus-scheduled concurrency test confirms default config
admits all 12 calls in both cases. The legacy parity test also confirms that
`NANOBOT_MAX_CONCURRENT_REQUESTS=2` caps the whole-turn gate while mapping a
per-lane limit of 2.

The existing `AgentLoop` whole-turn semaphore remains unchanged until P09.
`NANOBOT_MAX_CONCURRENT_REQUESTS` also seeds the scheduler's default provider
and model lane limit when its value is positive; an explicit scheduler setting applies when
the environment variable is unset, and zero leaves the scheduler uncapped.
With the environment variable set, the old whole-turn limit and the new
per-model-call lane limit both apply. The legacy gate limits whole turns while
the scheduler limit applies independently to each provider and model lane, so
workloads using multiple lanes can admit more model calls than the old setting
alone. This preserves the existing turn gate but does not claim exact parity
for environment-configured workloads. The default uncapped scheduler adds no
admission ceiling.
