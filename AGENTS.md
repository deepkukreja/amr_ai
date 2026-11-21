# Robot Buildworks project workflow

Use applicable domain skills under `.agents/skills`; each has a dated, verified baseline and concrete validation guidance. Inspect current source, Git state and active processes before acting. Continue this codebase and retain attribution, user edits, datasets, weights and backups.

Generic skill workflows do not authorize discarding existing work, replacing working dependencies, adding cloud/database providers or publishing changes. Use test-first development incrementally for new behavior; preserve prior implementations and run checks appropriate to the change. Keep training/inference/ROS environments separate and record actual measured results.

Unknown or stale measurements remain unknown. Keep simulation provenance explicit. Cloud, AI, voice and remote requests cannot bypass local command ownership and protective functions. The existing simulated stop is not independent emergency-stop hardware; simulation does not establish personnel safety.

## GitHub publishing

Apply the [AMR GitHub publishing skill](.agents/skills/amr-github-publish/SKILL.md) when completing verified repository changes. The user's standing preference across AMR chats is to commit the task's changes, push to the intended GitHub branch, and report the commit link.
