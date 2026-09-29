"""Automated operations over the document model: Generate, Improve, Simplify."""

from vectrify.operations import methods
from vectrify.operations.contract import (
    ACTIONS,
    Budget,
    Method,
    OperationRequest,
    OperationResult,
    Permissions,
    Proposal,
    RunContext,
    available,
    method,
    register,
)
from vectrify.operations.jobs import RESOURCES, Job

__all__ = [
    "ACTIONS",
    "RESOURCES",
    "Budget",
    "Job",
    "Method",
    "OperationRequest",
    "OperationResult",
    "Permissions",
    "Proposal",
    "RunContext",
    "available",
    "method",
    "methods",
    "register",
]
