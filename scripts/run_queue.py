"""Run the GPU stages of a queue config one after another: train, predict on the test split, evaluate.

Each stage is resumable: a finished stage is skipped, an interrupted training continues from its latest durable checkpoint,
an interrupted prediction continues after the rows already written. Progress goes to a log file and a state file, and a
heartbeat line with the stage's own ETA is written every few minutes. A stage that fails is retried once and then marked
failed; stages that need it are skipped and the queue continues with the others.

    uv run --no-sync python scripts/run_queue.py --config configs/queue_261102.json
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ["uv", "run", "--no-sync", "--project", "repositories/wbbs-vqa_vgrounding"]
ENVIRONMENT = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(ROOT / "repositories" / "wbbs-vqa_vgrounding" / "src")}


def path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


class Queue:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.log_dir = path(config["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.log_dir / "state.json"
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.exists() else {}
        self.log_file = self.log_dir / f"queue-{time.strftime('%y%m%d-%H%M%S')}.log"

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def mark(self, name: str, status: str) -> None:
        self.state[name] = {"status": status, "at": datetime.now().isoformat(timespec="seconds")}
        self.state_file.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def wait_for_marker(self) -> None:
        marker = self.config.get("wait_for_marker")
        if not marker:
            return
        target = path(marker["file"])
        while marker["text"] not in (target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""):
            self.log(f"waiting for {marker['text']} in {target.name}")
            time.sleep(marker["poll_seconds"])
        self.log("the scoring run finished")

    def check_resources(self) -> None:
        running = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True).stdout.lower()
        busy = sorted({name for name in self.config["blocked_processes"] if f'"{name.lower()}.exe"' in running})
        if busy:
            raise RuntimeError(f"apps that compete for RAM and the GPU are running: {', '.join(busy)}")
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.split()[0])
        if free < self.config["min_free_vram_mb"]:
            raise RuntimeError(f"the GPU has {free} MiB free, the stage needs {self.config['min_free_vram_mb']}")
        disk = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-PSDrive {str(ROOT)[0]}).Free / 1GB"], capture_output=True, text=True, check=True).stdout.strip()
        if float(disk) < self.config["min_free_disk_gb"]:
            raise RuntimeError(f"{float(disk):.1f} GB free on the project drive, the stage needs {self.config['min_free_disk_gb']}")

    def run(self, name: str, command: list[str]) -> bool:
        stage_log = self.log_dir / f"{name}-{time.strftime('%y%m%d-%H%M%S')}.log"
        self.log(f"{name}: {' '.join(command)}")
        with stage_log.open("wb") as handle:
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=ENVIRONMENT)
            last = time.time()
            while process.poll() is None:
                time.sleep(30)
                if time.time() - last >= self.config["heartbeat_seconds"]:
                    last = time.time()
                    self.log(f"{name}: {self.progress(stage_log)}")
        self.log(f"{name}: exit code {process.returncode}, log {stage_log.name}")
        return process.returncode == 0

    @staticmethod
    def progress(stage_log: Path) -> str:
        tail = stage_log.read_bytes()[-4000:].decode("utf-8", errors="ignore").replace("\r", "\n").splitlines()
        lines = [line.strip() for line in tail if "eta=" in line or "ETA" in line or "rows_per_second" in line]
        return (lines[-1] if lines else "starting")[:220]

    def stage(self, spec: dict) -> None:
        name = spec["name"]
        if self.state.get(name, {}).get("status") == "done":
            self.log(f"{name}: already done")
            return
        if spec.get("requires") and self.state.get(spec["requires"], {}).get("status") != "done":
            self.log(f"{name}: skipped, {spec['requires']} did not finish")
            self.mark(name, "skipped")
            return
        if spec.get("not_after_local") and datetime.now() > datetime.fromisoformat(spec["not_after_local"]):
            self.log(f"{name}: dropped, it is later than {spec['not_after_local']}")
            self.mark(name, "dropped")
            return
        for attempt in range(1 + self.config["retries"]):
            try:
                self.check_resources()
                ok = {"train": self.train, "predict": self.predict, "score": self.score, "command": self.command}[spec["kind"]](spec)
            except RuntimeError as error:
                self.log(f"{name}: {error}")
                ok = False
            if ok:
                self.mark(name, "done")
                return
            self.log(f"{name}: attempt {attempt + 1} did not finish")
            time.sleep(60)
        self.mark(name, "failed")

    def train(self, spec: dict) -> bool:
        done, run_dir = path(spec["done_file"]), path(spec["run_dir"])
        if done.exists():
            return True
        command = PROJECT + ["qwen3vl-train", "--config", spec["config"]]
        if run_dir.exists():
            if (run_dir / "resume" / "latest.json").exists():
                command.append("--resume")
            else:
                run_dir.rename(run_dir.with_name(run_dir.name + time.strftime(".failed-%y%m%d-%H%M%S")))
        return self.run(spec["name"], command) and done.exists()

    def score(self, spec: dict) -> bool:
        return self.run(spec["name"], PROJECT + ["python", "-m", "training.score", "--config", spec["score_config"], "--model", spec["model"], "--split", spec["split"]])

    def command(self, spec: dict) -> bool:
        return self.run(spec["name"], spec["command"])

    def predict(self, spec: dict) -> bool:
        out = path(spec["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        predictions = out / "predictions.jsonl"
        count = sum(1 for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()) if predictions.exists() else 0
        if count < spec["rows"]:
            command = PROJECT + ["qwen3vl-predict", "--config", spec["config"], "--adapter", spec["adapter"], "--input", spec["input"], "--output", str(predictions), "--max-new-tokens", "96", "--batch-size", str(spec["batch_size"])]
            if count:
                command.append("--resume")
            if not self.run(spec["name"], command):
                return False
        if not (out / "metrics.json").exists():
            if not self.run(spec["name"] + "-metrics", PROJECT + ["wbbs-evaluate-qwen3", "--task", "multitask", "--expected", spec["expected"], "--predictions", str(predictions), "--output", str(out / "metrics.json")]):
                return False
        if not (out / "audit.json").exists():
            return self.run(spec["name"] + "-audit", PROJECT + ["python", "-m", "analysis.qwen_multitask", "--expected", spec["expected"], "--predictions", str(predictions), "--output", str(out / "audit.json")])
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    queue = Queue(json.loads(args.config.read_text(encoding="utf-8")))
    queue.log(f"queue start, stages {[s['name'] for s in queue.config['stages']]}, state {queue.state}")
    queue.wait_for_marker()
    for spec in queue.config["stages"]:
        queue.stage(spec)
    queue.log(f"queue finished: {json.dumps({k: v['status'] for k, v in queue.state.items()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
