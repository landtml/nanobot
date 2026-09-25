# Observational Memory golden fixtures

`nanobot/agent/observational_memory` is a Python port of Mastra's Observational
Memory as released in `@mastra/memory@1.1.0`, the version behind Mastra's
LongMemEval result. The tests in `tests/agent/observational_memory/test_golden.py`
hold the port to that release byte for byte: every prompt, message formatter,
output parser, context renderer and token count is compared with what the
upstream TypeScript produces for the same inputs.

The golden files are committed, so the tests need neither Node nor the upstream
source. Regenerate them only when `fixtures.py` changes:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    --branch @mastra/memory@1.1.0 https://github.com/mastra-ai/mastra.git /tmp/mastra
git -C /tmp/mastra sparse-checkout set packages/memory/src/processors/observational-memory

python scripts/om_golden/build_harness.py \
    /tmp/mastra/packages/memory/src/processors/observational-memory
python scripts/om_golden/fixtures.py
(cd scripts/om_golden && npm install)
for tz in UTC America/Los_Angeles Asia/Shanghai; do
    TZ=$tz node --experimental-strip-types scripts/om_golden/generate.mjs
done
```

`build_harness.py` refuses upstream files whose SHA-256 does not match 1.1.0.
Node 22.6 or newer is required for `--experimental-strip-types`.
