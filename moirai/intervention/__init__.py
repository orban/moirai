"""Causal intervention study harness.

Implements the engineering-pilot infrastructure for the comparative selector
study described in ``docs/specs/2026-09-05-intervention-spec-v0.3.md``:
provenance recovery, eligibility screening, checkpoint cloning, forced-action
injection, blocked randomization, an immutable trial ledger, a hard spend cap,
and a fixed-weight analysis. Nothing in this package launches paid inference by
itself; a runtime adapter must be supplied explicitly.
"""
