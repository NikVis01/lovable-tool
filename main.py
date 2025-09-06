import os
import requests
from pathlib import Path
from dotenv import load_dotenv
import re

# Load .env from project root (if present)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

api_key = os.getenv("LANGFLOW_API_KEY")
run_url = os.getenv("LANGFLOW_RUN_URL", "http://localhost:7860/api/v1/run/6e0c7f49-5b1d-4674-8343-43a1d6cf142f")
if "?stream=" not in run_url:
	run_url = run_url + ("&" if "?" in run_url else "?") + "stream=false"
if not api_key:
	raise SystemExit("LANGFLOW_API_KEY not set. Add it to .env or export before running.")

user_query = Path("search_query.txt").read_text(encoding="utf-8")
print(user_query)

payload = {"output_type": "chat", "input_type": "chat", "input_value": user_query}
headers = {"Content-Type": "application/json", "x-api-key": api_key}

HEADER_A = "rank,profile_url,name,headline,location,primary_stack,score,evidence_sources"
HEADER_B = "name,headline,linkedin_url,github_url,affiliations,primary_stack,location,education,cold_dm"


def extract_text(obj: dict, fallback: str) -> str:
	for path in (
		["outputs", 0, "outputs", 0, "results", "text", "data", "text"],
		["outputs", 0, "outputs", 0, "results", "message", "data", "text"],
		["artifacts", "text", "repr"],
		["artifacts", "message"],
	):
		cur = obj
		ok = True
		for key in path:
			try:
				cur = cur[key]
			except Exception:
				ok = False
				break
		if ok and isinstance(cur, str) and cur.strip():
			return cur
	return fallback


def find_header(text: str) -> tuple[str, int]:
	i_a = text.lower().find(HEADER_A)
	i_b = text.lower().find(HEADER_B)
	if i_a != -1 and (i_b == -1 or i_a < i_b):
		return HEADER_A, i_a
	if i_b != -1:
		return HEADER_B, i_b
	return "", -1


def parse_fixed_columns(segment: str, num_cols: int) -> list[list[str]]:
	rows: list[list[str]] = []
	i = 0
	N = len(segment)
	while i < N:
		while i < N and segment[i] in " \t\r\n":
			i += 1
		if i >= N:
			break
		fields: list[str] = []
		for col in range(num_cols):
			if i < N and segment[i] == '"':
				i += 1
				buf = []
				while i < N:
					ch = segment[i]
					if ch == '"':
						if i + 1 < N and segment[i + 1] == '"':
							buf.append('"')
							i += 2
							continue
						else:
							i += 1
							break
					else:
						buf.append(ch)
						i += 1
				field = "".join(buf)
			else:
				buf = []
				while i < N and segment[i] not in [',', '\n', '\r']:
					buf.append(segment[i])
					i += 1
				field = "".join(buf).strip()
			fields.append(field)
			if col < num_cols - 1 and i < N and segment[i] == ',':
				i += 1
		if len(fields) == num_cols:
			rows.append(fields)
		else:
			break
		j = i
		while j < N and segment[j] in " \t\r\n":
			j += 1
		if j < N and segment[j] == '}':
			break
		i = j
	seen = set()
	dedup: list[list[str]] = []
	for r in rows:
		key = (r[0] or "").strip().lower()
		if key in seen:
			continue
		seen.add(key)
		dedup.append(r)
	return dedup[:50]


def fix_existing_file(path: Path) -> None:
	if not path.exists():
		return
	raw = path.read_text(encoding="utf-8")
	# Normalize header spacing like: "education, cold_dm" -> "education,cold_dm"
	raw = re.sub(r",\s*cold_dm\b", ",cold_dm", raw, flags=re.IGNORECASE)
	# Insert newline after closing quote that ends a record when a new name starts
	raw = re.sub(r'"\s+(?=[A-Z][^,]+,https?://|[A-Z][^,]+,Null)', '"\n', raw)
	path.write_text(raw, encoding="utf-8")


try:
	resp = requests.post(run_url, json=payload, headers=headers, timeout=90)
	resp.raise_for_status()
	data = resp.json()
	text = extract_text(data, resp.text)
	header, idx = find_header(text)
	if idx != -1:
		segment = text[idx + len(header):]
		num_cols = len(header.split(','))
		rows = parse_fixed_columns(segment, num_cols)
		if rows:
			csv_lines = [header] + [",".join([f.replace("\n", " ").replace("\r", " ") for f in r]) for r in rows]
			csv_text = "\n".join(csv_lines) + "\n"
			out_dir = Path(__file__).parent / "src" / "data"
			out_dir.mkdir(parents=True, exist_ok=True)
			out_file = out_dir / "agent_output.csv"
			out_file.write_text(csv_text, encoding="utf-8")
			print(f"Wrote CSV -> {out_file}")
		else:
			print("Parser produced no rows; leaving file untouched")
	else:
		print("Header not found in response; attempting to fix existing file")
		fix_existing_file(Path(__file__).parent / "src" / "data" / "agent_output.csv")
except Exception as e:
	print(f"Parse error: {e}")
