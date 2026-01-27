# SnookerVision
Official repository for the SnookerVisionaries.

Backend-focused starter structure for a snooker table system with camera ball tracking and projection guidance.

## Objectives
- Detect and track snooker balls in real time from an overhead camera.
- Project guidance cues for ball repositioning and shot assistance.
- Provide a clean backend foundation for calibration, tracking, and projection logic.

## Scope
- Camera ingest and tracking pipeline.
- Projection output system.
- Calibration and shared utilities.

## Out of scope (for now)
- Robotic ball placement.
- Voice recognition.
- Commercial-grade hardware integration.

## Repo layout
- `backend/camera`: Camera input and frame capture adapters.
- `backend/tracking`: Ball detection and tracking logic.
- `backend/projection`: Projection mapping and output control.
- `backend/shared`: Common utilities and shared types.
- `backend/config`: Config and runtime settings.
- `docs`: Project documentation.
- `scripts`: Helper scripts for setup or tooling.
- `tests`: Test scaffolding for backend modules.

## Making changes to the repo
1. Create a branch for your change.
2. Keep changes small and focused.
3. Write clear commit messages (what + why).
4. Update `README.md` or `docs/` when behavior or structure changes.
5. Open a PR or share the branch for review before merging.
