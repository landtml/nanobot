// Record golden outputs from the Mastra Observational Memory 1.1.0 harness.
//
//   python scripts/om_golden/build_harness.py <upstream observational-memory dir>
//   (cd scripts/om_golden && npm install)
//   TZ=UTC node --experimental-strip-types scripts/om_golden/generate.mjs
//
// Timestamps are formatted in the process timezone, exactly as Mastra did, so
// the generator runs once per timezone listed in TIMEZONES.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  buildObserverSystemPrompt,
  buildObserverPrompt,
  buildMultiThreadObserverPrompt,
  formatMessagesForObserver,
  parseObserverOutput,
  parseMultiThreadObserverOutput,
  optimizeObservationsForContext,
} from './harness/observer-agent.ts';
import {
  buildReflectorSystemPrompt,
  buildReflectorPrompt,
  parseReflectorOutput,
} from './harness/reflector-agent.ts';
import { TokenCounter } from './harness/token-counter.ts';
import { Engine, CONTINUATION_REMINDER, addRelativeTimeToObservations } from './harness/engine.ts';

const here = dirname(fileURLToPath(import.meta.url));
const goldenDir = join(here, '..', '..', 'tests', 'agent', 'observational_memory', 'golden');
const fixtures = JSON.parse(readFileSync(join(goldenDir, 'fixtures.json'), 'utf8'));
const tz = process.env.TZ;
if (!tz) throw new Error('set TZ explicitly');

const conv = name => fixtures.conversations[name];
const threadMap = threads => new Map(Object.entries(threads).map(([id, name]) => [id, conv(name)]));
const counter = new TokenCounter();

const out = { timezone: tz };

out.system_prompts = {
  observer: buildObserverSystemPrompt(false),
  observer_multi_thread: buildObserverSystemPrompt(true),
  reflector: buildReflectorSystemPrompt(),
};
out.continuation_reminder = CONTINUATION_REMINDER;

out.formatted = Object.fromEntries(
  Object.keys(fixtures.conversations).map(name => [name, formatMessagesForObserver(conv(name))]),
);
out.formatted_truncated = fixtures.max_part_length.map(c =>
  formatMessagesForObserver(conv(c.conversation), { maxPartLength: c.max }),
);
out.observer_prompts = fixtures.observer_prompts.map(c =>
  buildObserverPrompt(c.existing ?? undefined, conv(c.conversation)),
);
out.multi_thread_prompts = fixtures.multi_thread_prompts.map(c =>
  buildMultiThreadObserverPrompt(c.existing ?? undefined, threadMap(c.threads), c.order),
);
out.reflector_prompts = fixtures.reflector_prompts.map(c =>
  buildReflectorPrompt(c.observations, c.manual ?? undefined, c.retry),
);

const plain = value => (value === undefined ? null : value);
out.observer_outputs = fixtures.observer_outputs.map(o => {
  const r = parseObserverOutput(o);
  return { observations: r.observations, currentTask: plain(r.currentTask), suggested: plain(r.suggestedContinuation) };
});
out.reflector_outputs = fixtures.reflector_outputs.map(o => {
  const r = parseReflectorOutput(o);
  return { observations: r.observations, suggested: plain(r.suggestedContinuation) };
});
out.multi_thread_outputs = fixtures.multi_thread_outputs.map(o => {
  const r = parseMultiThreadObserverOutput(o);
  return [...r.threads.entries()].map(([id, t]) => ({
    id,
    observations: t.observations,
    currentTask: plain(t.currentTask),
    suggested: plain(t.suggestedContinuation),
  }));
});
out.optimize = fixtures.optimize.map(o => optimizeObservationsForContext(o));
out.relative_time = fixtures.relative_time.map(c => addRelativeTimeToObservations(c.observations, new Date(c.now)));

const engine = new Engine();
out.context = fixtures.context.map(c =>
  engine.formatObservationsForContext(
    c.observations,
    c.current_task ?? undefined,
    c.suggested ?? undefined,
    c.unobserved ?? undefined,
    c.now ? new Date(c.now) : undefined,
  ),
);
out.thread_ids = await Promise.all(fixtures.thread_ids.map(id => engine.representThreadIDInContext(id)));
out.thread_sections = [];
for (const c of fixtures.thread_sections) {
  const section = await engine.wrapWithThreadTag(c.thread, c.observations);
  const existing = c.existing.replaceAll('THREAD', await engine.representThreadIDInContext(c.thread));
  out.thread_sections.push({ section, merged: engine.replaceOrAppendThreadSection(existing, c.thread, section) });
}
out.token_strings = fixtures.token_strings.map(s => counter.countString(s));
out.token_messages = Object.fromEntries(
  Object.keys(fixtures.conversations).map(name => [
    name,
    { each: conv(name).map(m => counter.countMessage(m)), total: counter.countMessages(conv(name)) },
  ]),
);
out.selection = fixtures.selection.map(c =>
  new Engine(c.messageTokens, 40_000, c.maxTokensPerBatch).selectAndBatch(threadMap(c.threads)) ?? null,
);
out.unobserved_blocks = await Promise.all(
  fixtures.unobserved_blocks.map(c => engine.formatUnobservedContextBlocks(threadMap(c.threads), c.current)),
);
out.thresholds = fixtures.thresholds.map(c => {
  const e = new Engine(c.messageTokens, c.observationTokens);
  return {
    dynamic: e.calculateDynamicThreshold(c.messageTokens, c.current),
    shouldReflect: e.shouldReflect(c.current),
  };
});

const slug = tz.replaceAll('/', '_');
writeFileSync(join(goldenDir, `golden.${slug}.json`), JSON.stringify(out, null, 2) + '\n');
console.log(`wrote golden.${slug}.json`);
