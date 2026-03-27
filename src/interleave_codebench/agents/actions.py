from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ActionKind = Literal["read", "shell", "edit", "test", "finish"]


@dataclass(slots=True)
class AgentAction:
    kind: ActionKind
    description: str
    command: str | None = None
    path: str | None = None
    old_text: str | None = None
    new_text: str | None = None

    def fingerprint(self) -> str:
        return "|".join(
            [
                self.kind,
                self.command or "",
                self.path or "",
                self.old_text or "",
                self.new_text or "",
            ]
        )

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)

