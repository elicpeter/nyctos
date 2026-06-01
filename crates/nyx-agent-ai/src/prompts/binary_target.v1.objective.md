Pentest the configured local binary target.

RUN
- run_id: @@RUN_ID@@
- project_id: @@PROJECT_ID@@

TARGET
@@TARGET@@

ARGV TEMPLATE
@@ARGV_TEMPLATE@@

SOURCE / INSPECTION LEADS
@@KNOWN_LEADS@@

ARTIFACT DIRECTORY
@@ARTIFACT_DIR@@

OPERATING NOTES
- The target is hostile native code; expect and hunt for memory
  corruption. Every execution goes through `sandbox.exec` only.
- Stage malformed inputs with `sandbox.write_file` and reference them via
  `@FILE:<name>` in the exec args.
- Stop after roughly @@MAX_TURNS@@ tool turns. Record only crashes or
  anomalies a `sandbox.exec` reproduced, with a byte-exact repro.
- Reply with exactly one JSON tool object per turn and nothing else.
