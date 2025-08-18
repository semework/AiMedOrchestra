# Compatibility shim: keep old imports working
# Allows: `from aimedorchestraagent import aimedorchestraAgent`
# but delegates to the canonical package path.
from aimedorchestra.orchestrator.agent import aimedorchestraAgent  # noqa: F401
