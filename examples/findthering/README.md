# Find the Ring

A 90-second browser game built on graphlift's out-of-fold scores.

```bash
# 1. export a Syn Env graph (from your Syn Env checkout)
python examples/synenv/export_synenv.py --mode balanced --seed 2 --out runs/synenv_balanced_s2
# 2. audit it (writes graphlift_report/scores.csv)
graphlift audit runs/synenv_balanced_s2
# 3. cut three boards and build the page
python examples/findthering/build_boards.py runs/synenv_balanced_s2 --boards 3 --out boards.json
python -c "d=open('boards.json').read().replace('</','<\\\\/'); t=open('examples/findthering/template.html').read(); open('find-the-ring.html','w').write(t.replace('__DATA__', d))"
```

`template.html` is a page body for a hosted artifact. To serve it as a
standalone site, wrap it in `<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"></head><body>…</body></html>`.
