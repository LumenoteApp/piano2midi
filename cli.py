"""Command line: python cli.py song.mp3 [--extract] [-o outdir]"""

import argparse
from pathlib import Path

import engine


def main() -> None:
    ap = argparse.ArgumentParser(description="Piano audio to MIDI (Transkun V2).")
    ap.add_argument("input", type=str, help="audio/video file or a YouTube link")
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument("--extract", action="store_true", help="isolate piano from a full mix first")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--min-velocity", type=int, default=0)
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--no-pedal", action="store_true", help="leave CC64 out of the MIDI")
    args = ap.parse_args()

    raw = str(args.input)
    if raw.startswith(("http://", "https://")):
        outdir = args.outdir or Path("output")
        src, title = engine.download_audio(raw, outdir / "_download")
        name = title
    else:
        outdir = args.outdir or Path(raw).parent
        src = Path(raw); name = src.stem
    base = outdir / name
    if args.extract:
        src = engine.separate_piano(src, outdir / f"{name}_piano.wav", args.device)
    t = engine.transcribe(src, args.device)
    t = engine.clean(t, engine.Cleanup(min_velocity=args.min_velocity,
                                       min_duration=args.min_duration,
                                       sustain_pedal=not args.no_pedal))
    for kind, path in engine.export(t, base).items():
        print(f"{kind:8s} -> {path}")


if __name__ == "__main__":
    main()
