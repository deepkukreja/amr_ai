# Robot Buildworks perception console

- This directory is the Windows-side development increment for the operator perception view. The WSL `rbot`, `amr_ai` and `amr_platform` worktrees own their existing implementations.
- Document source provenance and clock domains. Keep missing, invalid and stale observations distinguishable from fresh ones. Unknown occupancy remains unknown.
- Keep the browser and demonstration bridge read-only. Changes that could publish navigation or motion requests belong in the locally supervised robotics stack and require a coordinated integration.
- Test changes against the browser boundary and against the source contracts they affect. Use deterministic, explicitly labeled simulation fixtures.
