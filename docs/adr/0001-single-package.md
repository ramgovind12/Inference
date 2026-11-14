# ADR 0001: One Python package for all components

Status: Accepted

## Context
InferScale has a Gateway, Queue, Scheduler, Router and Worker. They could be
separate packages or one package with one subpackage per component.

## Decision
Use one package, `inference`, with a subpackage per component. Shared schemas
and errors live in `inference.core`. Components depend on `core` and never on
each other except through interfaces (for example `RequestQueue`,
`RequestRouter`, `Backend`).

## Consequences
- One lockfile, one test run, and simple refactoring.
- Shared schemas cannot drift between components.
- The same image can run every role with a different command.
- The worker image contains code it never runs. Split the package if the
  worker needs dependencies that conflict with the rest, for example vLLM.
