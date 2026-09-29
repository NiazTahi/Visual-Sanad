import csv
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(p.open(encoding="utf-8-sig"))) if p.exists() else []
print(
    "rows",
    len(rows),
    "errors",
    sum(1 for r in rows if r.get("error")),
    "unparsed",
    sum(1 for r in rows if r.get("prediction") == "UNPARSED"),
)
