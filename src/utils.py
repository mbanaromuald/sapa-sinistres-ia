"""
Fonctions utilitaires transverses : journalisation locale (audit trail) et
petites aides de formatage.

Principe de confidentialité : le journal d'audit n'enregistre JAMAIS le
contenu intégral des documents ou des données personnelles des assurés. Il
trace uniquement l'action effectuée (type d'opération, horodatage, nombre
de caractères traités, modèle utilisé) afin de permettre une traçabilité
sans exposer d'information sensible dans les logs.
"""
import datetime as _dt
import json
from pathlib import Path

from src.config import LOGS_DIR

AUDIT_LOG_PATH = Path(LOGS_DIR) / "audit.log"


def log_action(action: str, details: dict | None = None) -> None:
    """Ajoute une ligne JSON au journal d'audit local (append-only)."""
    entry = {
        "horodatage": _dt.datetime.now().isoformat(timespec="seconds"),
        "action": action,
        "details": details or {},
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_recent_logs(n: int = 20) -> list[dict]:
    """Retourne les n dernières entrées du journal d'audit."""
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def format_fcfa(montant: float) -> str:
    """Formate un montant en francs CFA avec séparateurs de milliers."""
    try:
        return f"{montant:,.0f} FCFA".replace(",", " ")
    except (TypeError, ValueError):
        return str(montant)
