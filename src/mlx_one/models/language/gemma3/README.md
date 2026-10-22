# Gemma3 text inference

`google/gemma-3-1b-it` at `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
is `EXPERIMENTAL_SUPPORTED` under the [functional model admission policy](../../../../../docs/model-admission.md).
The tested primary configuration is stored BF16 weights, batch size one, native dense
execution, FP16 attention KV and bounded sliding state. Python/CLI/HTTP/SSE have bounded
functional smoke evidence; no practical maximum context or other-hardware certification
is claimed.

Known limitations: historical sliding-window boundary comparisons showed numerical
differences; repeated or weak answers were observed. These are disclosed quality/research
limitations and do not reinstate the retired numerical admission gates. Text checkpoint
smoke does not certify Gemma vision or arbitrary precision conversions. Historical direct
128/16 performance is available in the [benchmark evidence](../../../../../docs/benchmarks/README.md);
128/128 results are pending. Performance does not establish instruction or answer quality.
