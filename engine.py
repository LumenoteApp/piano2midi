"""
Piano audio -> MIDI engine.

Pipeline
  1. (optional) BS-Roformer-SW isolates the piano stem from a full mix.
  2. Transkun V2 transcribes the piano. Its checkpoint is trained WITHOUT
     pedal-extended notes, so note lengths are real key presses, and the
     sustain / sostenuto / soft pedals come out as separate events.
  3. We write MIDI:
       *_keys.mid      key-press lengths + CC64/66/67 pedal (what a player did)
       *_sustain.mid   notes stretched to where the pedal lets them ring
                       (for tools that ignore CC64), pedal CC kept as well
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import warnings

import numpy as np

# torch checkpoint / transkun emit noisy UserWarnings on every segment
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

Log = Callable[[str], None]

PEDAL_CC = {-64: 64, -66: 66, -67: 67}  # transkun encodes pedals as negative pitch


@dataclass
class Note:
    pitch: int
    start: float
    end: float
    velocity: int


@dataclass
class Pedal:
    cc: int          # 64 sustain, 66 sostenuto, 67 soft
    start: float
    end: float
    value: int = 127


@dataclass
class Transcription:
    notes: list[Note]
    pedals: list[Pedal]
    duration: float
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- audio

def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("FFmpeg was not found on PATH. Install it (winget install ffmpeg).")
    return exe


def load_audio(path: Path, sr: int, channels: int = 2) -> np.ndarray:
    """Decode any audio/video file to float32 (frames, channels) at `sr`."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg could not read the file: " + proc.stderr.decode(errors="ignore")[-400:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio.reshape(-1, channels).copy()


def to_wav(src: Path, dest: Path, sr: int = 44100) -> Path:
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_f32le", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed: " + proc.stderr[-400:])
    return dest


def download_audio(url: str, folder: Path, log: Log = print) -> tuple[Path, str]:
    """Download a YouTube (or any yt-dlp supported) link as MP3. Returns (path, title)."""
    import re
    import sys
    folder.mkdir(parents=True, exist_ok=True)
    log("Downloading audio with yt-dlp…")
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
           "-x", "--audio-format", "mp3", "--audio-quality", "0",
           "--ffmpeg-location", str(Path(ffmpeg_exe()).parent),
           "-o", str(folder / "source.%(ext)s"),
           "--print", "after_move:title", "--", url]
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False, env=env)
    mp3 = folder / "source.mp3"
    if proc.returncode != 0 or not mp3.exists():
        err = [l for l in (proc.stderr or proc.stdout).splitlines() if "ERROR" in l]
        raise RuntimeError((err[-1] if err else (proc.stderr or "yt-dlp failed")[-300:]).strip())
    title = (proc.stdout.strip().splitlines() or ["download"])[-1]
    title = re.sub(r'[\\/:*?"<>|]+', "_", title).strip(" .")[:120] or "download"
    log(f"Downloaded: {title}")
    return mp3, title


# --------------------------------------------------------------------------- device

def pick_device(requested: str = "auto") -> str:
    import torch
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- stems

def separate_piano(src: Path, out_wav: Path, device: str = "auto", log: Log = print) -> Path:
    """Isolate the piano stem with BS-Roformer-SW (6-stem model, has a piano stem)."""
    import torch
    dev = pick_device(device)
    if dev == "cuda":
        dev = "cuda:0"
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="p2m-stems-") as tmp:
        tmp = Path(tmp)
        work_in, work_out = tmp / "in", tmp / "out"
        work_in.mkdir()
        work_out.mkdir()
        to_wav(src, work_in / "mix.wav")
        log(f"Separating piano stem with BS-Roformer-SW on {dev} (first run downloads ~700 MB)…")
        manifest = None
        try:
            from bs_roformer import BSRoformerSession
            with BSRoformerSession(device=dev, progress=False) as session:
                manifest = session.infer(str(work_in), store_dir=str(work_out), verbose=False)
        except ImportError:
            _separate_legacy(work_in, work_out, dev)
        piano = _find_piano(work_out, getattr(manifest, "outputs", None))
        if piano is None:
            names = ", ".join(p.name for p in work_out.rglob("*.wav")) or "nothing"
            raise RuntimeError(f"Separator produced no piano stem (got: {names}).")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(piano, out_wav)
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    log(f"Piano stem ready in {time.time() - t0:.1f}s")
    return out_wav


def _separate_legacy(work_in: Path, work_out: Path, dev: str) -> None:
    """bs-roformer-infer <= 0.1.5 has only the folder API."""
    from argparse import Namespace
    import torch
    import yaml
    from ml_collections import ConfigDict
    from bs_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config
    from bs_roformer.inference import SafeLoaderWithTuple, run_folder

    ckpt, cfg_path = ensure_model_assets(DEFAULT_MODEL)
    with open(cfg_path, encoding="utf-8") as fh:
        config = ConfigDict(yaml.load(fh, Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    target = torch.device(dev)
    model = model.to(target).eval()
    run_folder(model, Namespace(input_folder=str(work_in), store_dir=str(work_out)),
               config, target, verbose=False)
    del model


def _find_piano(folder: Path, outputs=None) -> Optional[Path]:
    for item in outputs or []:
        path = Path(getattr(item, "output_path", ""))
        if str(getattr(item, "output_id", "")).lower() == "piano" and path.exists():
            return path
    hits = sorted(folder.rglob("*piano*.wav"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- transkun

_model = None
_model_dev = None
_model_lock = threading.Lock()


def get_model(device: str = "auto"):
    """Load Transkun V2 once and keep it in memory."""
    global _model, _model_dev
    dev = pick_device(device)
    with _model_lock:
        if _model is not None and _model_dev == dev:
            return _model
        import importlib.resources as res
        import moduleconf
        import torch
        pre = res.files("transkun") / "pretrained"
        conf_mgr = moduleconf.parseFromFile(str(pre / "2.0.conf"))
        TransKun = conf_mgr["Model"].module.TransKun
        model = TransKun(conf=conf_mgr["Model"].config).to(dev)
        ckpt = torch.load(str(pre / "2.0.pt"), map_location=dev, weights_only=False)
        state = ckpt.get("best_state_dict", ckpt.get("state_dict"))
        model.load_state_dict(state, strict=False)
        model.eval()
        _model, _model_dev = model, dev
        return model


def transcribe(src: Path, device: str = "auto", log: Log = print) -> Transcription:
    import torch
    model = get_model(device)
    dev = _model_dev
    audio = load_audio(src, sr=int(model.fs), channels=2)
    dur = len(audio) / model.fs
    log(f"Transcribing {dur:.1f}s of audio with Transkun V2 on {dev}…")
    t0 = time.time()
    with torch.no_grad():
        x = torch.from_numpy(audio).to(dev)
        raw = model.transcribe(x, discardSecondHalf=False)
    notes, pedals = [], []
    for n in raw:
        if n.pitch > 0:
            notes.append(Note(int(n.pitch), float(n.start), float(n.end), int(n.velocity)))
        elif n.pitch in PEDAL_CC:
            pedals.append(Pedal(PEDAL_CC[n.pitch], float(n.start), float(n.end), int(n.velocity) or 127))
    notes.sort(key=lambda n: (n.start, n.pitch))
    pedals.sort(key=lambda p: (p.cc, p.start))
    if dev == "cuda":
        torch.cuda.empty_cache()
    log(f"Found {len(notes)} notes, {sum(p.cc == 64 for p in pedals)} sustain presses "
        f"({time.time() - t0:.1f}s)")
    return Transcription(notes, pedals, dur, {"device": dev})


# --------------------------------------------------------------------------- cleanup

@dataclass
class Cleanup:
    min_velocity: int = 0          # drop ghost notes quieter than this (stems can leak)
    min_duration: float = 0.0      # drop blips shorter than this (seconds)
    sustain_pedal: bool = True     # include CC64 in the output
    other_pedals: bool = True      # include sostenuto (66) / soft (67)
    velocity_scale: float = 1.0    # global velocity multiplier
    quantize_pedal: float = 0.0    # ignore pedal lifts shorter than this (re-pedal flicker)


def clean(t: Transcription, c: Cleanup) -> Transcription:
    notes = [Note(n.pitch, n.start, n.end, max(1, min(127, round(n.velocity * c.velocity_scale))))
             for n in t.notes
             if n.velocity >= c.min_velocity and (n.end - n.start) >= c.min_duration]
    pedals = []
    for cc in (64, 66, 67):
        if cc == 64 and not c.sustain_pedal:
            continue
        if cc != 64 and not c.other_pedals:
            continue
        group = sorted((p for p in t.pedals if p.cc == cc), key=lambda p: p.start)
        merged: list[Pedal] = []
        for p in group:
            if merged and p.start - merged[-1].end < c.quantize_pedal:
                merged[-1].end = max(merged[-1].end, p.end)
            else:
                merged.append(Pedal(p.cc, p.start, p.end, p.value))
        pedals += merged
    return Transcription(notes, pedals, t.duration, dict(t.info))


def sustain_extend(notes: list[Note], pedals: list[Pedal]) -> list[Note]:
    """Stretch each note to where the sustain pedal would let it ring.

    A note released while CC64 is down keeps sounding until the pedal lifts,
    or until the same key is struck again (the damper/hammer cuts it).
    """
    sus = sorted((p for p in pedals if p.cc == 64), key=lambda p: p.start)
    starts = np.array([p.start for p in sus]) if sus else np.zeros(0)
    by_pitch: dict[int, list[Note]] = {}
    for n in sorted(notes, key=lambda n: n.start):
        by_pitch.setdefault(n.pitch, []).append(n)

    out = []
    for pitch, seq in by_pitch.items():
        for i, n in enumerate(seq):
            end = n.end
            if len(sus):
                k = int(np.searchsorted(starts, n.end, side="right")) - 1
                if k >= 0 and sus[k].start <= n.end < sus[k].end:
                    end = max(end, sus[k].end)
            if i + 1 < len(seq):
                end = min(end, max(n.end, seq[i + 1].start - 0.005))
            out.append(Note(pitch, n.start, end, n.velocity))
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


# --------------------------------------------------------------------------- midi

def write_midi(path: Path, notes: list[Note], pedals: list[Pedal], tempo_bpm: float = 120.0) -> Path:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=tempo_bpm)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        if n.end - n.start < 0.01:
            n = Note(n.pitch, n.start, n.start + 0.01, n.velocity)
        inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                           start=float(n.start), end=float(n.end)))
    for p in pedals:
        inst.control_changes.append(pretty_midi.ControlChange(p.cc, int(p.value), float(p.start)))
        inst.control_changes.append(pretty_midi.ControlChange(p.cc, 0, float(p.end)))
    inst.control_changes.sort(key=lambda c: c.time)
    pm.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    return path


def export(t: Transcription, out_base: Path) -> dict[str, Path]:
    """Write both MIDI flavours. Returns {'keys': path, 'sustain': path}."""
    keys = write_midi(out_base.with_name(out_base.name + "_keys.mid"), t.notes, t.pedals)
    sustained = sustain_extend(t.notes, t.pedals)
    sus = write_midi(out_base.with_name(out_base.name + "_sustain.mid"), sustained, t.pedals)
    return {"keys": keys, "sustain": sus}


def to_json(t: Transcription) -> dict:
    return {
        "duration": t.duration,
        "notes": [[n.pitch, round(n.start, 4), round(n.end, 4), n.velocity] for n in t.notes],
        "sustained": [[n.pitch, round(n.start, 4), round(n.end, 4), n.velocity]
                      for n in sustain_extend(t.notes, t.pedals)],
        "pedals": [[p.cc, round(p.start, 4), round(p.end, 4)] for p in t.pedals],
        "info": t.info,
    }
