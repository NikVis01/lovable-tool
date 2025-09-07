### Lovable Candidate Knowledge Graph

Build and visualize a candidate knowledge graph backed by Neo4j + GDS with sentence-transformer embeddings and clean community detection. See similary between candidates and generated ideal profiles based off of job opening descriptions.

image.png

### Prereqs
- Python 3.11+
- uv (`pip install uv`) - otherwise can use apt or package manager
- Docker with Docker Compose (recommended, for Neo4j + GDS)

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run scraping agent
```bash
uv run langflow run
python src/scrape/main.py
```

### Run simple analysis to check that csv has been generated
```bash
python src/scrape/analysis.py
```

Note: sometimes the agent tweaks out and mismatches csv headers, drop rows that cause problems.

### Start Neo4j (GDS)
```bash
python src/graph/setup_neo4j.py
```

### Run analysis
```bash
python src/graph/test_graph.py
```
Outputs:
- `candidate_knowledge_graph_gds.html` (interactive graph)
- Data written to Neo4j (Browser at `http://localhost:7474`)

### Tune communities (optional)
- Fewer, larger groups: lower edge `threshold`, raise `top_k`, or set `target=3/4` for Leiden.

### Config (env)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (see `.env` created by setup)


# TO DO #
1. Add column type and classes "Human" or "Generated".
2. (OPTIONAL) Unfuck pipeline agent -> graph.
3. Script to find people similar to imaginary engineers.