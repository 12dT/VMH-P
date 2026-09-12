# Checkpoint Artifacts

This directory contains layout documentation for the nine host/dataset settings
reported in the main table.

Large frozen-backbone checkpoints and cached host representations are not stored
in Git because several files exceed GitHub's normal file-size limits. They can
be restored locally under the same layout:

```text
artifacts/checkpoints/main_table_seed42/<host>/<dataset>/
  backbone/
  cache/
  vmhp/
```

The binary `vmhp/` checkpoint files are kept local and are not part of the public
Git artifact.
