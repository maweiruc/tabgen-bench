"""Append-only Core RunManifest snapshots and terminal publication, without schema changes."""

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distributions

import yaml

from tabgen_bench.core.config import FrozenConfig
from tabgen_bench.core.manifest import (
    CacheRecord,
    ContractHashes,
    EnvironmentInfo,
    RunManifest,
    VersionInfo,
)

from .artifacts import IntegrityError, atomic_write, digest, json_bytes


def utc_now():
    return datetime.now(UTC)


def environment_info():
    libraries = {d.metadata["Name"]: d.version for d in distributions()}
    return EnvironmentInfo(
        python=platform.python_version(),
        operating_system=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        actual_device="cpu",
        gpu=None,
        reproducibility_level="same-environment CPU; independence not established",
        libraries=tuple(VersionInfo(name=n, version=v) for n, v in sorted(libraries.items())),
    )


def git_info(root):
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


class ManifestJournal:
    def __init__(self, store, run_id, cell, *, command=(), git_root=".", resume_of=None):
        self.store, self.run_id, self.cell = store, run_id, cell
        self.artifacts, self.cache, self.events = [], [], []
        self.sequence = 0
        self.store.create(run_id)
        resolved = FrozenConfig.from_mapping({"cell": cell.model_dump(mode="json")})
        hashes = {node.id: node.key for node in cell.nodes}
        schema = cell.dataset.spec.dataset_schema
        commit, dirty = git_info(git_root)
        self.current = RunManifest(
            run_id=run_id,
            status="pending",
            created_at=utc_now(),
            command=tuple(command),
            git_commit=commit,
            git_dirty=dirty,
            environment=environment_info(),
            dataset_id=cell.dataset.spec.id,
            dataset_version=cell.dataset.spec.version,
            source=cell.dataset.spec.source,
            hashes=ContractHashes(
                data=hashes["data"],
                split=hashes["split"],
                schema_hash=schema.fingerprint(),
                generation_schema_view=schema.generation_view().fingerprint(),
                validity_rules=schema.validity_rules().fingerprint(),
                validity_rules_version=schema.constraint_rules_version,
                dependency_lock=cell.dependency_lock_hash,
            ),
            seeds=cell.seeds,
            generator_id=cell.generator.id,
            generator_version=cell.generator.version,
            generator_params=cell.generator.params,
            suite_id=cell.evaluation.id,
            suite_version=cell.evaluation.version,
            metric_versions=tuple(
                VersionInfo(name=m.id, version=m.version) for m in cell.evaluation.metrics
            ),
            resolved_config=resolved,
            config_hash=resolved.fingerprint(),
            n_synthetic=cell.n_synthetic,
        )
        self._snapshot()
        self.write(
            "config.resolved.yaml", yaml.safe_dump(resolved.to_dict(), sort_keys=True).encode()
        )
        self.write(
            "execution.json",
            json_bytes(
                {"storage_root": str(store.root), "resume_of": resume_of, "command": list(command)}
            ),
        )
        self.write("dataset_metadata.json", json_bytes(cell.dataset.provenance.to_dict()))
        self.current = self.current.model_copy(
            update={"status": "running", "started_at": utc_now()}
        )
        self._snapshot()

    def write(self, name, payload):
        artifact = self.store.write(self.run_id, name, payload)
        self.artifacts.append(artifact)
        return artifact

    def _snapshot(self):
        self.current = RunManifest.model_validate(
            {
                **self.current.model_dump(),
                "artifacts": tuple(self.artifacts),
                "cache": tuple(self.cache),
            }
        )
        payload = json_bytes(self.current.model_dump(mode="json"))
        folder = self.store.directory(self.run_id) / "manifests"
        name = f"{self.sequence:06d}"
        atomic_write(folder / f"{name}.json", payload)
        atomic_write(folder / f"{name}.sha256", (digest(payload) + "\n").encode())
        self.sequence += 1

    def stage(self, stage, key, hit, *, metadata=None, entry=None):
        event = {
            "event_version": "1.0.0",
            "run_id": self.run_id,
            "dataset_id": self.cell.dataset.spec.id,
            "generator_id": self.cell.generator.id,
            "replicate_seed": self.cell.seeds.replicate_seed,
            "stage": stage,
            "key": key,
            "cache_hit": hit,
            "time": utc_now().isoformat(),
            "status": "completed",
            "ended_at": utc_now().isoformat(),
            "metadata": metadata or {},
        }
        event["started_at"] = (metadata or {}).get("started_at")
        if metadata and metadata.get("status") == "failed":
            event["status"] = "failed"
        if entry is not None:
            event["cache_origin"] = entry.metadata
            event["cache_receipt_sha256"] = digest((entry.root / "entry.json").read_bytes())
            event["cache_artifacts"] = [a.model_dump() for a in entry.artifacts]
        self.events.append(event)
        self.write(f"checkpoints/{len(self.events):06d}.json", json_bytes(event))
        self.cache.append(CacheRecord(stage=stage, key=key, hit=hit))
        self._snapshot()

    def finish(self, status, *, error=None, metric_success_fraction=None):
        if self.current.status != "running":
            raise ValueError("terminal manifest is immutable")
        self.write("logs.jsonl", b"".join(json_bytes(e) for e in self.events))
        self.current = self.current.model_copy(
            update={
                "status": status,
                "ended_at": utc_now(),
                "metric_success_fraction": metric_success_fraction,
                "error_type": None if error is None else type(error).__name__,
                "error_message": None if error is None else str(error) or repr(error),
            }
        )
        self._snapshot()
        atomic_write(
            self.store.directory(self.run_id) / "manifest.json",
            json_bytes(self.current.model_dump(mode="json")),
        )
        return self.current


def load_manifest(store, run_id):
    folder = store.directory(run_id)
    snapshots = sorted((folder / "manifests").glob("*.json"))
    if not snapshots:
        raise IntegrityError("run has no committed manifest snapshot")
    latest = None
    for path in snapshots:
        payload = path.read_bytes()
        receipt = path.with_suffix(".sha256")
        # A crash between publishing the snapshot and its receipt leaves it uncommitted.
        if not receipt.exists():
            if path != snapshots[-1]:
                raise IntegrityError("missing intermediate manifest receipt")
            continue
        if digest(payload) != receipt.read_text().strip():
            raise IntegrityError("manifest snapshot hash mismatch")
        latest = RunManifest.model_validate_json(payload)
        if latest.run_id != run_id:
            raise IntegrityError("manifest run identity mismatch")
    if latest is None:
        raise IntegrityError("no complete manifest snapshot")
    terminal = folder / "manifest.json"
    if terminal.exists() and terminal.read_bytes() != json_bytes(latest.model_dump(mode="json")):
        raise IntegrityError("terminal manifest differs from immutable snapshot")
    store.verify(latest)
    return latest


def validate_checkpoints(store, manifest, cache):
    for artifact in manifest.artifacts:
        if artifact.path.startswith("checkpoints/"):
            event = json.loads(store.read(manifest.run_id, artifact))
            if "cache_receipt_sha256" not in event:
                continue
            stage = event["stage"].split(":", 1)[0]
            entry = cache.get(stage, event["key"])
            if (
                entry is None
                or digest((entry.root / "entry.json").read_bytes()) != event["cache_receipt_sha256"]
            ):
                raise IntegrityError("resume checkpoint cache receipt mismatch")
