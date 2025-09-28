from pathlib import Path
import json, sys
from jsonschema import validate, exceptions
from jsonschema.validators import Draft202012Validator

REC = Path("docs/receipt.latest.json")
SCH = Path("schema/receipt.v1.2.schema.json")

if not REC.exists():
    print("[err] docs/receipt.latest.json missing"); sys.exit(2)
if not SCH.exists():
    print("[err] schema/receipt.v1.2.schema.json missing"); sys.exit(2)

j = json.loads(REC.read_text("utf-8"))
s = json.loads(SCH.read_text("utf-8"))
try:
    validate(instance=j, schema=s, cls=Draft202012Validator)
except exceptions.ValidationError as e:
    print("[fail] schema:", e.message)
    print("path:", "/".join(map(str, e.path)))
    sys.exit(2)

print("[ok] v1.2 schema validation passed")
