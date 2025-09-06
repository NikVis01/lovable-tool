# Candidate Knowledge Graph with Neo4j GDS and Tensor Embeddings

This implementation creates a knowledge graph of candidates using Neo4j with Graph Data Science (GDS) plugin, tensor-based representations, and applies Leiden clustering to identify communities of similar candidates.

## Features

- **Neo4j Integration**: Uses Neo4j graph database for scalable graph storage and processing
- **Graph Data Science (GDS)**: Leverages Neo4j's GDS plugin for advanced graph algorithms
- **Tensor Embeddings**: Uses Sentence Transformers to create high-dimensional vector representations of candidate profiles
- **GDS Similarity Analysis**: Uses GDS FastRP and cosine similarity for efficient similarity computation
- **GDS Leiden Clustering**: Applies the GDS Leiden algorithm for community detection
- **Interactive Visualization**: Creates interactive HTML visualizations using Plotly
- **Community Analysis**: Provides detailed analysis of each community's characteristics
- **Docker Support**: Easy setup with Docker Compose for Neo4j with GDS plugin

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Neo4j with Graph Data Science plugin

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Neo4j with GDS plugin:
```bash
cd src/graph
python setup_neo4j.py
```

This will:
- Create a `.env` file with Neo4j configuration
- Start Neo4j container with GDS plugin
- Test the connection and GDS availability

## Usage

### Basic Usage

```python
from graph import CandidateKnowledgeGraph

# Initialize with your CSV file
csv_path = 'path/to/your/candidates.csv'
kg = CandidateKnowledgeGraph(csv_path)

# Run the complete analysis
results = kg.run_full_analysis()
```

### Step-by-Step Usage

```python
# Connect to Neo4j
kg.connect_neo4j()

# Load and preprocess data
kg.load_data()

# Create tensor embeddings
kg.create_tensor_embeddings()

# Load candidates into Neo4j
kg.load_candidates_to_neo4j()

# Create GDS graph projection
kg.create_gds_graph()

# Compute similarity using GDS
kg.compute_similarity_with_gds()

# Apply Leiden clustering using GDS
kg.apply_leiden_clustering_gds()

# Analyze communities
community_stats = kg.analyze_communities()

# Create visualizations
graph_fig = kg.visualize_graph('graph.html')
heatmap_fig = kg.create_community_heatmap('heatmap.png')

# Clean up
kg.close_neo4j()
```

### Running the Test Script

```bash
cd src/graph
python test_graph.py
```

## Output Files

The analysis generates several output files:

1. **candidate_knowledge_graph_gds.html**: Interactive network visualization showing:
   - Nodes representing candidates
   - Edges representing similarity relationships
   - Color-coded communities from GDS Leiden clustering
   - Hover information with candidate details

2. **community_similarity_heatmap_gds.png**: Heatmap showing similarity between communities

3. **Neo4j Database**: The candidates and relationships are stored in Neo4j for further analysis

## Data Requirements

The CSV file should contain the following columns:
- `Name`: Candidate name
- `Tech stack`: Technologies used (semicolon-separated)
- `Affiliations`: Organizations/affiliations (semicolon-separated)
- `Most Notable Company`: Previous notable company
- `Coolest Problem`: Description of interesting problems worked on
- `Degree`: Educational background
- `Location`: Geographic location
- `Years Experience`: Years of professional experience

## Algorithm Details

### Tensor Embeddings
- Uses the `all-MiniLM-L6-v2` Sentence Transformer model
- Combines multiple text fields (tech stack, affiliations, company, problems, degree)
- Creates 384-dimensional vector representations
- Stores embeddings as node properties in Neo4j

### GDS Similarity Computation
- Uses Neo4j GDS FastRP algorithm for efficient embedding processing
- Computes cosine similarity between candidate embeddings
- Creates `SIMILAR_TO` relationships with similarity scores
- Applies threshold (default: 0.3) to filter relationships

### GDS Leiden Clustering
- Uses Neo4j GDS Leiden algorithm for community detection
- Leverages relationship weights (similarity scores) for clustering
- Stores community assignments as node properties
- Provides scalable community detection for large graphs

### Community Analysis
- Analyzes common technologies within communities
- Identifies shared affiliations
- Computes average experience levels
- Provides geographic distribution

## Customization

### Adjusting Similarity Threshold
```python
# In compute_similarity_matrix method
threshold = 0.4  # Increase for fewer connections, decrease for more
```

### Changing Embedding Model
```python
# In create_tensor_embeddings method
self.model = SentenceTransformer('all-mpnet-base-v2')  # Larger, more accurate model
```

### Modifying Community Detection
```python
# In apply_leiden_clustering method
partition = leidenalg.find_partition(g_ig, leidenalg.RBERVertexPartition)  # Different algorithm
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- neo4j: Neo4j database driver
- matplotlib/seaborn: Static visualizations
- plotly: Interactive visualizations
- torch: Tensor operations
- sentence-transformers: Text embeddings
- scikit-learn: Machine learning utilities
- python-dotenv: Environment variable management
- requests: HTTP requests for setup

## Example Output

The analysis will output community information like:

```
Community 0 (3 members):
  Members: Filip Larsson, Ammar Alzeno, Matei Cananau
  Top Tech: Python(3), Java(2), Go(2)
  Top Affiliations: KTH(3), Natively(1)
  Avg Experience: 3.3 years
  Locations: Stockholm
```

This helps identify groups of candidates with similar technical backgrounds, affiliations, and experience levels.
