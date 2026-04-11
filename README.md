# ai-detector

Minimal FastAPI scaffold named `ai-detector`.

Quick start:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run.sh
```
Local swagger is here: http://127.0.0.1:8000/docs

App module: `app.main` (FastAPI instance `app`).

**Project Layout AI-Generated**

ai-dector/
- app/
	- main.py           # FastAPI application instance and router includes
	- core/             # configuration, logging, dependency overrides
	- common/           # shared utilities, exceptions, auth helpers
	- schemas/          # shared Pydantic request/response models
	- services/         # shared service helpers (clients, model wrappers)
	- features/         # vertical slices for each detection feature
		- image/          # image detection router, schemas, service, tests
			- api.py
			- schemas.py
			- service.py
			- tests/
		- video/          # video detection router, job handling, workers
		- document/       # document detection/extraction router and logic
- requirements.txt    # Python dependencies
- run.sh              # run helper (uvicorn app.main:app)
- .gitignore          # ignored files
- README.md           # this file

Notes on responsibilities
- `app/main.py`: imports and mounts routers, registers middleware and events.
- `features/*/api.py`: expose HTTP endpoints (APIRouter) for the slice.
- `features/*/service.py`: business logic, model inference, batching, queuing.
- `features/*/schemas.py`: Pydantic schemas local to the slice.
- `core/`: central config (env, secrets), metrics, and dependency wiring.
- `common/`: cross-cutting helpers (error handlers, security, storage clients).

Design guidance
- Use vertical slices (one folder per feature) so each feature contains its HTTP surface, validation, and business logic.
- Keep model/GPU-heavy inference in `service.py` but consider offloading to workers or separate services for production.
- Add `tests/` per slice and integration tests in the project `tests/` directory.
