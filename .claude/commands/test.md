Run tests on the Raven HPC cluster. Arguments are passed through to pytest.

Usage: /test [pytest args]

Examples:
- /test                           -- run all tests
- /test -k test_encoder           -- run only encoder tests
- /test -k test_environment -v    -- run environment tests verbosely
- /test --sync-only               -- just sync files, don't run tests

$ARGUMENTS

Run the remote test script:

```bash
scripts/remote_test.sh $ARGUMENTS
```

If the SSH connection check fails, tell the user they need to run `ssh raven` in a separate terminal first.
If tests fail, read the log at `.claude/test-logs/latest.log` for detailed output and help debug.
