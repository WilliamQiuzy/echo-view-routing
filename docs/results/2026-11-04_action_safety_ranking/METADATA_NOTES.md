# Sealed-schema clarifications

The sealed metric/cache/summary bytes remain unchanged.

- `action_statistics_source_role.whole_axis_policy_cap: false` and the corresponding full-task field mean there is no blanket prohibition on an interval spanning the entire axis. They do **not** describe the number of edits. Every A61 E/N/C policy selects at most one canonical action (or no-op) per sequence; G separately preserves the whole-axis occupancy majority. Candidate/accepted interval and changed-position fractions and whole-axis counts are exported explicitly.
- The inherited row key `full_system_inference_seconds` is cached-feature/head/policy time. It excludes frozen A08/JEPA producer inference, as `cached_heads_do_not_include_A08_JEPA_producer_latency` states. All A61 policies require both teachers operationally. Physical early-producer GPU/CPU cost and read-only adoption cost are reported separately, once per actual stage/attempt rather than once per logical policy.
- The 1/2/8 structural cost ceilings count source-clip pieces under constant-family native references and label-preserving transforms. They are not an inspection of actual GT label runs or model performance. The cost harness generates its own reference arrays, retaining the original metadata-parsing exception.
- Reported H/R and safety constraints refer to propagated clip-family reference labels. They are not expert frame-visibility or clinical safety labels. C is an empirical reused-inner calibration rule, without a safety guarantee on outer streams.
