# Robot Buildworks project workflow

Use applicable domain skills under `.agents/skills`; each has a dated, verified baseline and concrete validation guidance. Inspect current source, Git state and active processes before acting. Continue this codebase and retain attribution, user edits, datasets, weights and backups.

Generic skill workflows do not authorize discarding existing work, replacing working dependencies, adding cloud/database providers or publishing changes. Use test-first development incrementally for new behavior; preserve prior implementations and run checks appropriate to the change. Keep training/inference/ROS environments separate and record actual measured results.

Unknown or stale measurements remain unknown. Keep simulation provenance explicit. Cloud, AI, voice and remote requests cannot bypass local command ownership and protective functions. The existing simulated stop is not independent emergency-stop hardware; simulation does not establish personnel safety.

## GitHub publishing

Apply the [AMR GitHub publishing skill](.agents/skills/amr-github-publish/SKILL.md) when completing verified repository changes. The user's standing preference across AMR chats is to commit the task's changes, push to the intended GitHub branch, and report the commit link.

## Cross-chat coordination

Apply the [AMR chat coordination skill](.agents/skills/amr-chat-coordination/SKILL.md) before planning or resuming substantial work. Read relevant peer updates and use the shared AMR project's local `coordination/` directory. When working in WSL, resolve the same Windows workspace through its mounted path. Check for file, interface, and runtime overlap before editing. Keep chat handoffs out of the repository and publish verified project changes through the GitHub workflow above.

## Integration and product theme

Apply the [AMR integration verification skill](.agents/skills/amr-integration-verification/SKILL.md) when a change crosses an AMR component boundary or presents evidence of integrated behavior.

Use the website's Robot Buildworks theme across final AMR interfaces. The [shared product theme](design/README.md) defines the design tokens and how to apply them. Keep test fixtures, temporary demos, and diagnostic tools functional and lightweight. Spend visual polish and review time on final deliverables.
