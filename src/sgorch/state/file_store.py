from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from ..logging_setup import get_logger
from .base import DeploymentSnapshot, StateStore


logger = get_logger(__name__)


class FileStateStore(StateStore):
    """JSON file-backed state store.

    Structure on disk:
    {
      "deployments": {
         "<name>": { "name": "<name>", "workers": [...], "allocated_ports": [...] }
      }
    }
    """

    def __init__(self, file_path: Optional[str] = None):
        # Precedence: explicit arg > env var > CWD default > home cache
        env_path = os.environ.get("SGORCH_STATE_FILE")
        if file_path:
            chosen = file_path
        elif env_path:
            chosen = env_path
        else:
            cwd_default = os.path.join(os.getcwd(), ".sgorch-state.json")
            chosen = cwd_default
        # Fallback: home cache if CWD not writable
        self.path = Path(chosen)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            fallback = Path(os.path.join(os.path.expanduser("~"), ".cache", "sgorch", "state.json"))
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.path = fallback
        self._lock = Lock()

        # Initialize file if missing
        if not self.path.exists():
            self._write_all({"deployments": {}})
            logger.info(f"Created new state file at {self.path}")

    def load_deployment(self, name: str) -> Optional[DeploymentSnapshot]:
        with self._lock:
            data = self._read_all()
            dep = data.get("deployments", {}).get(name)
            if not dep:
                return None
            try:
                return DeploymentSnapshot.from_dict(dep)
            except Exception as e:
                logger.warning(f"Failed to parse snapshot for {name}: {e}")
                return None

    def save_deployment(self, snapshot: DeploymentSnapshot) -> None:
        with self._lock:
            data = self._read_all()
            deployments: Dict = data.setdefault("deployments", {})
            deployments[snapshot.name] = snapshot.to_dict()
            self._write_all(data)

    def delete_deployment(self, name: str) -> None:
        with self._lock:
            data = self._read_all()
            deployments: Dict = data.setdefault("deployments", {})
            if name in deployments:
                del deployments[name]
                self._write_all(data)

    def _read_all(self) -> Dict:
        if not self.path.exists():
            return {"deployments": {}}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"State file corrupted, resetting: {self.path}")
            return {"deployments": {}}

    def _write_all(self, obj: Dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, self.path)
