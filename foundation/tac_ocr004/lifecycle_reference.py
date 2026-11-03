"""TAC-OCR-004 necessary lifecycle gate; not the neural forward graph."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Binding:
    signature: str
    generation: str
    owners: tuple[str, ...]

def accepts(current: Binding, references: Binding, state: Binding) -> bool:
    """Only the ownership/signature projection; shapes/timing remain source gates."""
    return (current.generation == references.generation == state.generation
            and current.owners == references.owners == state.owners
            and current.signature == state.signature)

def renew(current: Binding, fresh_generation: str) -> Binding:
    """Freshness is an explicit premise, not a UUID collision-proof claim."""
    if not fresh_generation or fresh_generation == current.generation:
        raise ValueError('fresh generation premise required')
    return Binding(current.signature, fresh_generation, current.owners)
