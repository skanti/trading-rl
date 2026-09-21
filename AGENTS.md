# Process ownership

Keep experiment and research source code inside this repository, under `research/`.
Save generated run summaries, CSVs, and charts under `/tmp` by default, not in
the code folder. Candidate run results belong in `/tmp/trading-backtests/candidate/`.
Do not create sibling clones or external worktrees unless the user explicitly
requests an external location.

The user manages all long-running processes in this repository, including live
trading and the dashboard/digest publisher.

- Do not start, restart, or create daemon/background processes unless the user
  explicitly asks for that process action. Code changes, deployment requests, and
  requests to make a change take effect do not authorize daemon starts or restarts.
- If a change requires a restart, tell the user which process needs restarting
  and leave the restart to them. Do not ask for routine restart permission.
- Do not stop or kill user-managed processes unless explicitly requested.
- When asked to stop processes started by an agent, verify process identity first;
  a newer instance of the same command may have been started by the user.
- Finite commands such as tests, builds, deployments, and explicitly requested
  test emails may run normally; they must not leave a daemon running.
