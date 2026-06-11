"""Quick probe: pre-holdout regime label series from the daily cache (TB-007 role 6)."""
import json, glob, os, collections

P6 = "/Users/drbretto/Desktop/Projects/trader-bot/runs/pkt_tb_006_clean_sheet_brain/prototype"
dirs = sorted(glob.glob(os.path.join(P6, "cache/s3/daily/*/inference.json")))
print("files:", len(dirs))
labs = {}
for p in dirs:
    d = os.path.basename(os.path.dirname(p))
    if d >= "2026-03-11":
        continue
    try:
        j = json.load(open(p))
        r = j.get("regime", {})
        lab = r.get("regime_label") or r.get("label")
        if lab:
            labs[d] = lab
    except Exception as e:
        print("ERR", d, e)
print("n labels pre-holdout:", len(labs))
ds = sorted(labs)
flips = [(ds[i], labs[ds[i - 1]], labs[ds[i]]) for i in range(1, len(ds))
         if labs[ds[i]] != labs[ds[i - 1]]]
print("n flips:", len(flips))
for f in flips:
    print(f)
print(collections.Counter(labs.values()))
json.dump(labs, open(os.path.dirname(os.path.abspath(__file__)) + "/regime_labels_preholdout.json", "w"), indent=0)
