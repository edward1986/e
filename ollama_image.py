import os
import subprocess
from pathlib import Path


def ensure_model_available(model_name="x/flux2-klein"):
    result = subprocess.run(
        ["ollama", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if model_name not in result.stdout:
        pull = subprocess.run(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"Failed to pull model: {pull.stderr.strip()}")


def generate_image_ollama(
    prompt: str,
    output_path: str,
    model: str = "x/flux2-klein",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
):
    ensure_model_available(model)

    out_dir = Path(output_path).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    before = set(Path.cwd().glob("*.png")) | set(Path.cwd().glob("*.jpg")) | set(Path.cwd().glob("*.jpeg"))

    cmd = ["ollama", "run", model]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    commands = [
        f"/set parameter num_outputs 1",
        f"/set parameter width {width}",
        f"/set parameter height {height}",
    ]

    if seed is not None:
        commands.append(f"/set parameter seed {seed}")

    commands.append(prompt)

    stdin_text = "\n".join(commands) + "\n"
    stdout, stderr = proc.communicate(stdin_text)

    if proc.returncode != 0:
        raise RuntimeError(f"Ollama failed: {stderr.strip() or stdout.strip()}")

    after = set(Path.cwd().glob("*.png")) | set(Path.cwd().glob("*.jpg")) | set(Path.cwd().glob("*.jpeg"))
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)

    if not new_files:
        raise RuntimeError("No generated image file found.")

    newest = new_files[0]
    target = Path(output_path).resolve()
    newest.replace(target)

    return str(target)
