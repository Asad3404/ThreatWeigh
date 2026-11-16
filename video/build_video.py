"""
Assemble the ThreatWeigh demo video from the three Playwright recordings.

    node video/record.js video/raw          # record (needs the deck + app servers, see video/README.md)
    python video/build_video.py             # -> video/threatweigh_demo.mp4

The app recording contains two ~30 s waits while Integrated Gradients runs for
8 models on a CPU. Those segments (timestamps logged by record.js in
app_marks.json) are sped up 8x and labelled on screen, so nothing is hidden.
"""
import glob
import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "threatweigh_demo.mp4")
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Medium.otf"
SPEED = 8
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "22", "-r", "30", "-an"]


def webm(part: str) -> str:
    return glob.glob(os.path.join(RAW, part, "*.webm"))[0]


def run(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def segment(src, start, end, out, speed=1.0, label=None):
    vf = [f"setpts=(PTS-STARTPTS)/{speed}"]
    if label:
        vf.append(f"drawtext=fontfile={FONT}:text='{label}':x=(w-tw)/2:y=24:fontsize=24:fontcolor=white:"
                  "box=1:boxcolor=0x0f1420@0.85:boxborderw=14")
    run(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src, "-vf", ",".join(vf), *ENC, out])


def main():
    marks = json.load(open(os.path.join(RAW, "app_marks.json")))
    t = {}
    for m in marks:
        t.setdefault(m["label"], []).append(m["t"])
    app = webm("app")
    tmp = tempfile.mkdtemp()
    parts = []

    def add(*a, **k):
        out = os.path.join(tmp, f"{len(parts):02d}.mp4")
        segment(*a, out=out, **k)
        parts.append(out)

    d1 = webm("deck1")
    add(d1, 0.4, float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                 "-of", "csv=p=0", d1])) - 0.2)
    label = f">> {SPEED}x fast-forward - Integrated Gradients on 8 models takes ~30 s on a CPU"
    cur = t["start"][0]
    for ws, we in zip(t["wait_start"], t["wait_end"]):
        add(app, cur, ws)
        add(app, ws, we, speed=SPEED, label=label)
        cur = we
    add(app, cur, t["end"][0])
    d2 = webm("deck2")
    add(d2, 0.4, float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                 "-of", "csv=p=0", d2])) - 0.2)

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", OUT])
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", OUT]))
    print(f"wrote {OUT}  ({dur:.1f} s, {os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
