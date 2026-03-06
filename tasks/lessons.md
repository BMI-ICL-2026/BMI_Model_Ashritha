# Lessons Learned

## Project Structure
- **ALL MATLAB files (.m, .mat) must go inside `MATLAB files/` subfolder** — never in the BMI root.
- Assume MATLAB's working directory is `MATLAB files/` when writing paths.
- Don't create wrapper scripts that compute paths relative to the BMI root — keep it simple, everything in one folder.
- Watch for stale duplicate .m files in the current directory — they shadow addpath'd versions.
- After writing new decoder files, always check `MATLAB files/` root for old copies that would shadow `myTeam/`.

## Test Workflow
- User runs `run_test` from MATLAB command window (current dir = `MATLAB files/`).
- Functions are called WITH arguments by the test harness — they are not standalone scripts.
- The 3-output form of positionEstimator is needed for stateful decoders (Kalman filter).
