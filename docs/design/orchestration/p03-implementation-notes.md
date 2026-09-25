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
cancellation-safe lane queues and retries selection. This extends the phase
touch list at the fallback candidate seam because wrapping the entire fallback
chain as one lane would either queue before trying an available fallback or
misreport local saturation as a provider failure.

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
