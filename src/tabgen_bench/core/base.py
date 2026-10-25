"""Immutable, versioned JSON contracts."""

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True
    )

    def fingerprint(self) -> str:
        from .hashing import config_hash

        return config_hash(self.model_dump(mode="json"))
