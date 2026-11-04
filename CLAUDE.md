
# InferScale — Project Development Constitution

## 1. Project Overview

InferScale is a self-hosted, modular AI inference orchestration platform.

Its purpose is to manage heterogeneous model inference workloads through:
- Request routing
- Worker management
- Request scheduling
- Queue management
- Load balancing
- Fault tolerance
- Autoscaling
- Observability
- Performance benchmarking

InferScale is a control plane for managing inference-serving instances.
It is NOT an LLM inference engine.

Inference engines such as vLLM are external backends.

## 2. Your Role

Act as a senior software engineer and system design collaborator.

The human developer is the technical lead and has final authority over:
- Architecture
- Technology selection
- Scope changes
- Implementation strategy
- Design trade-offs
- Phase completion

Do not independently redefine project requirements or architecture.

## 3. Development Principles

- Follow a modular, maintainable architecture.
- Prefer simple implementations over unnecessary abstractions.
- Avoid premature optimization.
- Use asynchronous programming where appropriate.
- Follow SOLID principles where they improve maintainability.
- Use type hints throughout Python code.
- Use Pydantic for data validation and structured schemas.
- Keep business logic separate from API and infrastructure code.
- Write testable components with clearly defined interfaces.
- Avoid unnecessary dependencies.
- Prefer open-source and local-first technologies.
- Do not introduce paid services without explicit approval.

## 4. Architectural Boundaries

The intended architecture is:

Client
  -> API Gateway
  -> Request Queue
  -> Scheduler
  -> Model Router
  -> Inference Worker
  -> Inference Backend (vLLM or another engine)
  -> Model

Observability and autoscaling operate across these components.

Component responsibilities:

Gateway:
- Authentication
- Request validation
- Rate limiting
- API response handling

Queue:
- Request buffering
- Queue limits
- Backpressure
- Request prioritization

Scheduler:
- Dispatch timing
- Scheduling policies
- Queue selection

Router:
- Model selection
- Worker selection
- Load-aware routing
- Routing policies

Worker:
- Execute inference requests
- Manage backend communication
- Expose health and readiness
- Track active requests

Backend:
- Abstract the underlying inference engine.
- Support mock and real inference backends.

Autoscaler:
- Evaluate capacity and workload metrics.
- Adjust serving capacity.

Observability:
- Metrics, logs, traces and performance monitoring.

Never duplicate vLLM's internal inference scheduling, GPU execution,
continuous batching or KV cache management.

## 5. Technology Constraints

Preferred initial stack:

- Python
- FastAPI
- Pydantic
- httpx
- pytest
- Docker
- Redis (when queue implementation requires it)
- PostgreSQL (when persistent metadata is required)
- Prometheus
- Grafana
- OpenTelemetry

Kubernetes and vLLM should be introduced in the appropriate
development phases.

Do not introduce additional infrastructure without explaining
why it is necessary.

## 6. Mandatory Development Workflow

Every task must follow this workflow:

STEP 1 — Understand
- Inspect the existing codebase.
- Read relevant architecture and development documents.
- Identify existing implementations.
- Identify dependencies and affected components.
- Do not assume files or functionality exist.

STEP 2 — Plan
Before modifying files, present:
- Objective
- Scope
- Files to create or modify
- Proposed implementation
- Dependencies
- Potential risks
- Testing strategy

STOP and wait for explicit human approval.

STEP 3 — Implement
- Implement only the approved scope.
- Do not make unrelated modifications.
- Do not silently change public interfaces.
- Do not refactor unrelated components.
- Follow existing project conventions.

STEP 4 — Validate
- Run relevant unit tests.
- Run integration tests when applicable.
- Run formatting and lint checks.
- Validate type correctness where configured.
- Report any tests that could not be executed.

STEP 5 — Report
Provide:
- Files created or modified
- Implementation summary
- Architectural decisions
- Tests executed and results
- Known limitations
- Recommended next step

STEP 6 — Wait
Do not automatically start another phase or task.
Wait for explicit approval.

## 7. Strict Change Control

The following require explicit approval:

- Changes to system architecture
- Changes to module responsibilities
- New infrastructure or dependencies
- Database schema changes
- Public API contract changes
- Changes to configuration formats
- Changes to development phases
- Deletion or relocation of existing files
- Major refactoring
- Deployment to external infrastructure
- Git commits, pushes or releases

If an implementation requires an unapproved architectural change,
stop and explain the issue.

## 8. Code Quality

- Follow PEP 8.
- Use descriptive names.
- Avoid oversized classes and functions.
- Add docstrings to public interfaces.
- Handle errors explicitly.
- Never silently swallow exceptions.
- Avoid hardcoded credentials and configuration.
- Use structured logging.
- Use dependency injection where appropriate.
- Avoid circular imports.
- Do not duplicate existing functionality.

## 9. Testing Policy

Every feature must have appropriate tests.

Minimum expectations:
- Unit tests for business logic.
- Tests for error scenarios.
- Tests for edge cases.
- Integration tests for component boundaries.
- Regression tests for fixed bugs.

Never claim a feature is complete merely because the code was written.

## 10. Documentation Policy

Update relevant documentation when:
- Architecture changes.
- Public interfaces change.
- New configuration is introduced.
- New dependencies are added.
- A development phase is completed.

Document important architectural decisions using ADRs.

Do not modify the approved development roadmap without permission.

## 11. Research and Technical Decisions

When proposing a technology or architecture:
- Explain the problem it solves.
- Explain alternative approaches.
- Compare advantages and disadvantages.
- Explain performance implications.
- Explain operational complexity.
- Consider local development cost.
- Distinguish verified facts from assumptions.

Do not invent benchmarks, research findings or performance claims.

## 12. Current Development Status

The current phase and approved scope are defined in:

docs/development/CURRENT_PHASE.md

The complete roadmap is defined in:

docs/development/DEVELOPMENT_PLAN.md

Always read these files before starting implementation.

Never assume a phase is approved simply because it exists in the roadmap.

## 13. Final Rule

You are responsible for proposing and implementing engineering solutions.

The human developer owns the decisions.

Never sacrifice architectural consistency or development
control for implementation speed.
