---
name: amr-github-publish
description: Verify, commit, and push completed repository changes when working on the Robot Buildworks AMR project. Use for AMR code, tests, documentation, and shareable research results across chats.
---

# Publish AMR changes to GitHub

The user has given standing authorization to commit and push completed, verified AMR project changes to their intended GitHub repository. Apply this preference across AMR chats, subject to any narrower instruction in the current request.

- Find the actual checkout for the work and check its status, branch, and push destination. The perception and training repository is `deepkukreja/amr_ai` on GitHub. On this laptop its source checkout is in WSL. The Windows AMR workspace is also used for local artifacts.
- Run the checks appropriate to the change and review the staged diff. Stage only files that belong to the task. Put shareable source, documentation, tests, and aggregate experiment results in Git. Keep secrets, private images and webcam frames, datasets, checkpoints, and generated runs local.
- Commit with a descriptive message and push to the intended branch. Use the existing upstream for `amr_ai` work on `main` and respect any branch the user specifies. Resolve an unclear destination, missing remote, or protected branch before publishing. Keep unrelated working changes intact and avoid force pushes.
- Confirm the push succeeded and give the user the commit link. Report any verification failure or unresolved push blocker with the local commit ID when available.

This workflow applies when there is a project change to publish. Read-only work and local artifacts need no commit.
