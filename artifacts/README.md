# Local Artifacts

This directory is reserved for optional local experiment artifacts.

`checkpoints/main_table_seed42/` may contain the frozen backbone checkpoint, cached
host representations, VMH-P head weight, and selector file for each of the nine
host/dataset settings in the current main table. It is intentionally ignored by
Git because these files are large and may be governed by the licenses of the
original backbone or dataset providers.
