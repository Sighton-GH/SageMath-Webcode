# Codebase Changes Summary

Date range: 2026-02-06 to 2026-02-07

## Files added
- .dockerignore
- README.old.md (previous README backup)
- SETUP_GUIDE.md
- SETUP_LOG.md
- SETUP_RECOUNT.md
- CHANGES_SUMMARY.md
- f1tenth_gym_ros/wasd_teleop.py

## Files modified
- Dockerfile
  - Fixed nested gym path for pip install.
  - Enabled BuildKit cache for apt and pip.
  - Installed uv_build and f1tenth_gym in system Python.
- docker-compose.yml
  - Published port 8765 for Foxglove WebSocket.
- README.md
  - Rewritten with beginner-friendly setup instructions and troubleshooting.
- setup.py
  - Added console script: wasd_teleop.
- SETUP_LOG.md
  - Updated with final runtime commands and Foxglove connection notes.

## Functional changes
- Added WASD/arrow key teleop node with adjustable speed/steering.
- Foxglove WebSocket port (8765) is now exposed by default in docker-compose.
- Docker build is faster and more reliable due to caching and correct package paths.
