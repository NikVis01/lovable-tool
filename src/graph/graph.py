import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from community import (
    drop_gds_graph as drop_gds_graph_util,
    project_gds_graph_for_leiden as project_gds_graph_for_leiden_util,
    assign_mutual_knn_components,
    label_nodes_by_community as label_nodes_by_community_util,
    leiden_write_final,
    leiden_write_intermediate_target,
)
from collections import defaultdict
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/candidates.csv"
TEXT_COLS = ["Name","Title","Affiliations","Tech stack","Location","Most Notable Company","Coolest Problem","Degree"]
ID_COL = "id"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS", "Password"))

AUTH = (NEO4J_USER, NEO4J_PASS)

class CandidateKnowledgeGraph:
    def __init__(self, csv_path=None):
        """Initialize the knowledge graph with Neo4j connection."""
        self.driver = None
        self.df = None
        self.embeddings = None
        self.similarity_matrix = None
        self.communities = None
        self.model = None
        self.csv_path = csv_path or CSV_PATH
        self.threshold = float(os.getenv("COSINE_SIM_THRESHOLD", "0.6"))
        
    def connect_neo4j(self):
        """Connect to Neo4j database."""
        try:
            print(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}")
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=AUTH)
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j at {NEO4J_URI}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def close_neo4j(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")
    
    def load_and_prepare_data(self):
        """Load and prepare candidate data."""
        print("Loading candidate data...")
        self.df = pd.read_csv(self.csv_path)
        # Normalize and alias columns to expected canonical names
        self._ensure_canonical_columns()
        
        # Add ID column if not present
        if ID_COL not in self.df.columns:
            self.df[ID_COL] = np.arange(len(self.df))
        
        # Clean and preprocess the data
        for col in TEXT_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
                self.df[col] = self.df[col].replace({'Null': '', 'NULL': '', 'null': ''})
        
        # Clean Years Experience column
        if 'Years Experience' in self.df.columns:
            def clean_experience(value):
                if pd.isna(value) or value == '' or str(value).strip() == 'Unable to determine':
                    return 0.0
                try:
                    return float(str(value).strip())
                except (ValueError, TypeError):
                    return 0.0
            
            self.df['Years Experience'] = self.df['Years Experience'].apply(clean_experience)
        
        print(f"Loaded {len(self.df)} candidates")
        return self.df
    
    def create_embeddings(self):
        """Create embeddings for candidate data."""
        print("Creating embeddings...")
        # Combine available text columns (avoid KeyError if any missing)
        available = [c for c in TEXT_COLS if c in self.df.columns]
        if not available:
            # Fallback: use any string-like columns
            available = [c for c in self.df.columns if self.df[c].dtype == object]
        texts = self.df[available].astype(str).agg(" ".join, axis=1).tolist()
        
        # Load sentence transformer model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create embeddings
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        # Expose alias expected by tests
        self.tensor_embeddings = self.embeddings
        
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    # --------------------
    # Helpers
    # --------------------
    def create_vector_index(self, index_name: str = "candidate_embedding_index"):
        """Create Neo4j vector index for candidate embeddings (idempotent)."""
        dims = int(self.embeddings.shape[1]) if self.embeddings is not None else 384
        with self.driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (c:Candidate) ON (c.embedding)
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: 'cosine'
                }} }}
            """)
            print(f"✅ Vector index '{index_name}' ensured (dims={dims})")

    def drop_gds_graph(self, name: str):
        """Drop a GDS in-memory graph if it exists (delegated)."""
        drop_gds_graph_util(self.driver, name)

    def project_gds_graph_for_leiden(self, name: str = 'candidates_with_relationships'):
        """Create UNDIRECTED GDS graph projection using SIMILAR_TO(similarity) (delegated)."""
        project_gds_graph_for_leiden_util(self.driver, name)
        print(f"Created GDS graph '{name}' for Leiden")
    def _ensure_canonical_columns(self):
        """Map variant CSV headers to canonical names and backfill missing ones."""
        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower().strip() if ch.isalnum())

        colmap = {norm(c): c for c in self.df.columns}
        aliases = {
            'Title': ['title', 'headline', 'jobtitle', 'role', 'position'],
            'Type': ['type', 'candidate_type', 'candidate_class'],
            'Name': ['name', 'fullname', 'candidate', 'candidate_name'],
            'LinkedIn': ['linkedin', 'linkedinurl', 'linkedin_url', 'linkedIn', 'li', 'publiclink', 'public_link', 'linkedinprofile', 'linkedinprofileurl'],
            'Github': ['github', 'githuburl', 'github_url', 'gitHub', 'gh'],
            'Affiliations': ['affiliations', 'schoolaffiliation', 'school', 'organization', 'org'],
            'Tech stack': ['techstack', 'skills', 'stack', 'primarystack', 'primary_stack'],
            'Location': ['location', 'city', 'country'],
            'Most Notable Company': ['mostnotablecompany', 'company', 'employer'],
            'Coolest Problem': ['coolestproblem', 'summary', 'shortsummary', 'notes'],
            'Degree': ['degree', 'degreeyear', 'degreeyearsketch', 'education'],
            'Years Experience': ['yearsexperience', 'years_experience', 'experienceyears', 'years', 'exp']
        }
        # Create canonical columns if missing by mapping or filling empty
        for canon, variants in aliases.items():
            found = None
            for v in variants:
                if v in colmap:
                    found = colmap[v]
                    break
            if found is not None:
                if found != canon and canon not in self.df.columns:
                    self.df[canon] = self.df[found]
            elif canon not in self.df.columns:
                # backfill empty string except numeric experience
                self.df[canon] = '' if canon != 'Years Experience' else 0.0
    
    def store_candidates_in_neo4j(self):
        """Store candidates and embeddings in Neo4j."""
        print("Storing candidates in Neo4j...")
        
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create candidates with embeddings
            for idx, row in self.df.iterrows():
                embedding = self.embeddings[idx].tolist()
                
                query = """
                CREATE (c:Candidate {
                    type: $type,
                    id: $id,
                    name: $name,
                    title: $title,
                    linkedin: $linkedin,
                    github: $github,
                    tech_stack: $tech_stack,
                    affiliations: $affiliations,
                    location: $location,
                    years_experience: $years_experience,
                    company: $company,
                    problem: $problem,
                    degree: $degree,
                    embedding: $embedding
                })
                """
                
                session.run(query, {
                    'type': row.get('Type', ''),
                    'id': int(row[ID_COL]),
                    'name': row.get('Name', ''),
                    'title': row.get('Title', row.get('headline', row.get('Headline', ''))),
                    'linkedin': row.get('LinkedIn', row.get('linkedin_url', '')),
                    'github': row.get('Github', row.get('github_url', '')),
                    'tech_stack': row.get('Tech stack', row.get('primary_stack', '')),
                    'affiliations': row.get('Affiliations', ''),
                    'location': row.get('Location', ''),
                    'years_experience': row.get('Years Experience', row.get('years_experience', 0.0)),
                    'company': row.get('Most Notable Company', ''),
                    'problem': row.get('Coolest Problem', ''),
                    'degree': row.get('Degree', row.get('education', '')),
                    'embedding': embedding
                })
            
            print(f"Stored {len(self.df)} candidates in Neo4j")
    
    def create_gds_graph(self):
        """Create GDS graph projection."""
        print("Creating GDS graph projection...")
        
        with self.driver.session() as session:
            # Check if projection exists
            try:
                result = session.run("CALL gds.graph.exists('candidates')")
                exists = result.single()['exists']
                print(f"Graph projection 'candidates' exists: {exists}")
                
                if exists:
                    session.run("CALL gds.graph.drop('candidates')")
                    print("Dropped existing 'candidates' graph projection")
                else:
                    print("No existing 'candidates' projection found (this is normal for first run)")
            except Exception as e:
                print(f"Note: Could not check existing graph projection: {e}")
            
            # Create node projection (only numeric properties for GDS)
            print("Creating new 'candidates' graph projection...")
            session.run("""
                CALL gds.graph.project(
                    'candidates',
                    'Candidate',
                    '*',
                    {
                        nodeProperties: ['embedding', 'years_experience']
                    }
                )
            """)
            print("✅ GDS graph projection 'candidates' created successfully")
    
    def compute_similarity_with_gds(self):
        """Use GDS to compute similarity and create relationships."""
        print("Computing similarity using GDS...")
        
        
        # Use the embeddings we already created instead of GDS FastRP
        # This avoids the session issue with GDS projections
        print("Using pre-computed embeddings for similarity analysis...")
        
        # Compute cosine similarity directly from our embeddings
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        # Create relationships based on similarity threshold (using dataframe IDs)
        relationships_created = 0
        thr = float(self.threshold)
        with self.driver.session() as session:
            for i in range(len(self.df)):
                id_i = int(self.df.iloc[i][ID_COL])
                for j in range(i + 1, len(self.df)):
                    id_j = int(self.df.iloc[j][ID_COL])
                    similarity = self.similarity_matrix[i, j]
                    if similarity > thr:
                        session.run("""
                            MATCH (a:Candidate {id: $id1}), (b:Candidate {id: $id2})
                            MERGE (a)-[:SIMILAR_TO {similarity: $similarity, threshold: $threshold}]->(b)
                        """, {
                            'id1': id_i,
                            'id2': id_j,
                            'similarity': float(similarity),
                            'threshold': thr
                        })
                        relationships_created += 1
            print(f"Created {relationships_created} similarity relationships")
            return self.similarity_matrix

    def create_similarity_edges_topk(self, top_k: int = 2, bidirectional: bool = True):
        """Create SIMILAR_TO edges by top-k cosine similarity per node.

        - Uses dataframe IDs for stable matching.
        - Optionally writes reciprocal edges.
        """
        thr = float(self.threshold)
        print(f"Creating top-{top_k} similarity edges (threshold={thr})...")
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.embeddings)
        n = len(self.df)
        with self.driver.session() as session:
            created = 0
            for i in range(n):
                id_i = int(self.df.iloc[i][ID_COL])
                sims = self.similarity_matrix[i].copy()
                sims[i] = -1.0
                idxs = np.argsort(-sims)[:max(0, min(top_k, n - 1))]
                for j in idxs:
                    score = float(sims[j])
                    if score < thr:
                        continue
                    id_j = int(self.df.iloc[int(j)][ID_COL])
                    session.run(
                        """
                        MATCH (a:Candidate {id: $id1}), (b:Candidate {id: $id2})
                        MERGE (a)-[:SIMILAR_TO {similarity: $s, threshold: $t}]->(b)
                        """,
                        { 'id1': id_i, 'id2': id_j, 's': score, 't': thr }
                    )
                    created += 1
                    if bidirectional:
                        session.run(
                            """
                            MATCH (a:Candidate {id: $id1}), (b:Candidate {id: $id2})
                            MERGE (b)-[:SIMILAR_TO {similarity: $s, threshold: $t}]->(a)
                            """,
                            { 'id1': id_i, 'id2': id_j, 's': score, 't': thr }
                        )
                        created += 1
        print(f"Created {created} SIMILAR_TO relationships")

    # Removed apply_cosine_communities; use Leiden-based methods instead.

    def label_nodes_by_community(self, max_old_labels: int = 100):
        """Attach label Community{n} to each Candidate node based on c.community.

        Also attempts to remove a bounded set of old Community labels to avoid label buildup.
        """
        print("Updating node labels by community for Neo4j Browser coloring...")
        label_nodes_by_community_util(self.driver, self.df, self.communities, ID_COL, max_old_labels)

    def apply_mutual_knn_communities(self, k: int | None = None):
        """Split graph into communities using mutual k-NN graph components.

        - For each node, connect to its top-k most similar neighbors (excluding self).
        - Keep an undirected edge only if the relation is mutual (i in top-k of j, j in top-k of i).
        - Communities are connected components of this mutual k-NN graph.
        """
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.embeddings)
        n = len(self.df)
        if k is None:
            try:
                k = int(os.getenv("KNN_K", "3"))
            except ValueError:
                k = 3
        k = max(1, min(k, max(1, n - 1)))

        # For each node, get indices of top-k similar nodes
        topk = []
        for i in range(n):
            sims = self.similarity_matrix[i].copy()
            sims[i] = -1.0
            idxs = np.argsort(-sims)[:k]
            topk.append(set(int(j) for j in idxs))

        # Build mutual edges
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in topk[i]:
                if i in topk[j]:
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components
        communities = [-1] * n
        cid = 0
        for i in range(n):
            if communities[i] != -1:
                continue
            stack = [i]
            communities[i] = cid
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if communities[v] == -1:
                        communities[v] = cid
                        stack.append(v)
            cid += 1

        # Persist results via utility
        self.communities = assign_mutual_knn_components(self.driver, self.df, self.similarity_matrix, ID_COL, k)
        self.df['community'] = self.communities
        print(f"Assigned {len(set(self.communities))} mutual-kNN communities (k={k})")
        return self.communities
    
    def apply_leiden_clustering_gds(self, random_seed: int = 23, include_intermediate: bool = False):
        """Apply Leiden clustering using GDS. Optionally stream intermediate communities.

        Docs: https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/
        """
        print("Applying Leiden clustering using GDS...")
        
        with self.driver.session() as session:
            # Drop and recreate the Leiden projection
            self.drop_gds_graph('candidates_with_relationships')
            self.project_gds_graph_for_leiden('candidates_with_relationships')

            cfg = {
                'relationshipWeightProperty': 'similarity',
                'randomSeed': random_seed,
            }
            if include_intermediate:
                cfg['includeIntermediateCommunities'] = True

            # Stream results with original node ids resolved via gds.util.asNode(nodeId)
            result = session.run(
                "CALL gds.leiden.stream('candidates_with_relationships', $cfg) "
                "YIELD nodeId, communityId RETURN gds.util.asNode(nodeId).id AS candidate_id, communityId",
                { 'cfg': cfg }
            )
            comm_map = {}
            for record in result:
                comm_map[int(record['candidate_id'])] = int(record['communityId'])

            # Write to nodes
            for cand_id, community_id in comm_map.items():
                session.run(
                    "MATCH (c:Candidate {id: $id}) SET c.community = $community",
                    { 'id': cand_id, 'community': community_id }
                )

        # Mirror to dataframe order
        self.communities = []
        with self.driver.session() as session:
            for idx in range(len(self.df)):
                node_id = int(self.df.iloc[idx][ID_COL])
                rec = session.run(
                    "MATCH (c:Candidate {id: $id}) RETURN c.community AS community",
                    { 'id': node_id }
                ).single()
                self.communities.append(int(rec['community']) if rec and rec['community'] is not None else 0)
        self.df['community'] = self.communities
        print(f"Found {len(set(self.communities))} communities")
        return self.communities

    def apply_leiden_target_communities(self, target: int = 2, random_seed: int = 19) -> list[int]:
        """Choose an intermediate Leiden level closest to target community count.

        Uses includeIntermediateCommunities and maps nodeId -> original `Candidate.id` via gds.util.asNode.
        """
        print(f"Applying Leiden targeting ~{target} communities (choose intermediate level)...")
        self.communities = leiden_write_intermediate_target(self.driver, self.df, ID_COL, target, random_seed)
        self.df['community'] = self.communities
        print(f"Final communities: {len(set(self.communities))}")
        return self.communities
    
    def get_graph_data_for_visualization(self):
        """Extract graph data from Neo4j for visualization."""
        print("Extracting graph data for visualization...")
        
        with self.driver.session() as session:
            # Get nodes and their properties
            nodes_result = session.run("""
                MATCH (c:Candidate)
                RETURN c.id as id, c.name as name, c.title as title, c.type as type, c.community as community,
                       c.tech_stack as tech_stack, c.affiliations as affiliations,
                       c.years_experience as experience
            """)
            
            nodes = []
            for record in nodes_result:
                nodes.append({
                    'id': record['id'],
                    'name': record['name'],
                    'title': record['title'],
                    'type': record['type'],
                    'community': record['community'],
                    'tech_stack': record['tech_stack'],
                    'affiliations': record['affiliations'],
                    'experience': record['experience']
                })
            
            # Get relationships
            edges_result = session.run("""
                MATCH (a:Candidate)-[r:SIMILAR_TO]->(b:Candidate)
                RETURN a.id as source, b.id as target, r.similarity as similarity
            """)
            
            edges = []
            for record in edges_result:
                edges.append({
                    'source': record['source'],
                    'target': record['target'],
                    'similarity': record['similarity']
                })
            
            return nodes, edges
    
    def _build_community_color_map(self) -> dict:
        """Assign a distinct color for each unique community id."""
        if self.communities is None:
            return {}
        unique = sorted(set(self.communities))
        # Compose a large palette and extend if needed
        palette = (
            px.colors.qualitative.Dark24
            + px.colors.qualitative.Set3
            + px.colors.qualitative.Plotly
            + px.colors.qualitative.Prism
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Alphabet
        )
        if len(unique) > len(palette):
            # Generate additional hues if communities exceed palette size
            import colorsys
            extra = []
            for k in range(len(unique) - len(palette)):
                h = (k / max(1, len(unique)))
                r, g, b = [int(255 * v) for v in colorsys.hsv_to_rgb(h, 0.6, 0.95)]
                extra.append(f"rgb({r},{g},{b})")
            palette = palette + extra
        return {comm: palette[i] for i, comm in enumerate(unique)}

    def visualize_graph(self, save_path=None):
        """Create interactive visualization of the knowledge graph."""
        print("Creating graph visualization...")
        
        nodes, edges = self.get_graph_data_for_visualization()
        
        if not nodes:
            print("No nodes found for visualization")
            return None
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            # Find node positions (simple circular layout)
            source_node = next((n for n in nodes if n['id'] == edge['source']), None)
            target_node = next((n for n in nodes if n['id'] == edge['target']), None)
            
            if source_node and target_node:
                # Simple circular layout
                n_nodes = len(nodes)
                source_angle = 2 * np.pi * edge['source'] / n_nodes
                target_angle = 2 * np.pi * edge['target'] / n_nodes
                
                source_x = np.cos(source_angle)
                source_y = np.sin(source_angle)
                target_x = np.cos(target_angle)
                target_y = np.sin(target_angle)
                
                edge_x.extend([source_x, target_x, None])
                edge_y.extend([source_y, target_y, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_comms = []
        
        # Color only by type (HUMAN vs GENERATED)
        type_color_map = {
            'HUMAN': '#1f77b4',
            'GENERATED': '#ff7f0e'
        }
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            x = np.cos(angle)
            y = np.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            name = node['name']
            tech = node['tech_stack']
            node_type = (node.get('type') or '').upper() or 'HUMAN'
            title = node.get('title') or ''
            
            node_text.append(f"{name}<br>Title: {title}<br>Tech: {tech}<br>Type: {node_type}")
            node_comms.append(node_type)
            
            # Size based on number of connections
            connections = len([e for e in edges if e['source'] == node['id'] or e['target'] == node['id']])
            node_sizes.append(max(10, min(30, connections * 3)))
        
        # Build separate traces per community for distinct coloring and legend
        node_traces = []
        names = [node['name'] for node in nodes]
        unique_types = sorted(set(node_comms))
        for t in unique_types:
            idxs = [i for i,c in enumerate(node_comms) if c == t]
            node_traces.append(
                go.Scatter(
                    x=[node_x[i] for i in idxs],
                    y=[node_y[i] for i in idxs],
                    mode='markers+text',
                    name=f"{t}",
                    hoverinfo='text',
                    text=[names[i] for i in idxs],
                    textposition="middle center",
                    hovertext=[node_text[i] for i in idxs],
                    marker=dict(
                        size=[node_sizes[i] for i in idxs],
                        color=type_color_map.get(t, '#8888ff'),
                        line=dict(width=2, color='black')
                    )
                )
            )
        
        # Create the plot
        fig = go.Figure(data=[edge_trace] + node_traces,
                       layout=go.Layout(
                           title=dict(
                               text='Candidate Knowledge Graph (colored by Type)',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents number of connections. Color represents type.",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if save_path:
            fig.write_html(save_path)
            print(f"Graph saved to {save_path}")
        
        return fig
    
    def analyze_communities(self):
        """Analyze the characteristics of each community."""
        print("\nCommunity Analysis:")
        print("=" * 50)
        
        community_stats = []
        
        for community_id in sorted(set(self.communities)):
            community_df = self.df[self.df['community'] == community_id]
            
            # Get most common tech stacks
            tech_stacks = []
            for tech in community_df['Tech stack']:
                if pd.notna(tech) and tech.strip():
                    tech_stacks.extend([t.strip() for t in tech.split(';')])
            
            tech_counter = defaultdict(int)
            for tech in tech_stacks:
                tech_counter[tech] += 1
            
            most_common_tech = sorted(tech_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Get most common affiliations
            affiliations = []
            for aff in community_df['Affiliations']:
                if pd.notna(aff) and aff.strip():
                    affiliations.extend([a.strip() for a in aff.split(';')])
            
            aff_counter = defaultdict(int)
            for aff in affiliations:
                aff_counter[aff] += 1
            
            most_common_aff = sorted(aff_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Calculate average experience
            exp_values = community_df['Years Experience']
            exp_values = exp_values[exp_values > 0]
            avg_exp = exp_values.mean() if len(exp_values) > 0 else 0
            
            stats = {
                'community_id': community_id,
                'size': len(community_df),
                'members': community_df['Name'].tolist(),
                'most_common_tech': most_common_tech,
                'most_common_affiliations': most_common_aff,
                'avg_experience': avg_exp,
                'locations': community_df['Location'].unique().tolist()
            }
            
            community_stats.append(stats)
            
            print(f"\nCommunity {community_id} ({len(community_df)} members):")
            print(f"  Members: {', '.join(community_df['Name'].tolist())}")
            print(f"  Top Tech: {', '.join([f'{tech}({count})' for tech, count in most_common_tech])}")
            print(f"  Top Affiliations: {', '.join([f'{aff}({count})' for aff, count in most_common_aff])}")
            print(f"  Avg Experience: {avg_exp:.1f} years")
            print(f"  Locations: {', '.join(stats['locations'])}")
        
        return community_stats
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline using Neo4j and GDS."""
        print("Starting full knowledge graph analysis with Neo4j GDS...")
        print("=" * 60)
        
        try:
            # Connect to Neo4j
            if not self.connect_neo4j():
                print("Cannot proceed without Neo4j connection")
                return None
            
            # Load and prepare data
            self.load_and_prepare_data()
            
            # Create embeddings
            self.create_embeddings()
            
            # Store candidates in Neo4j
            self.store_candidates_in_neo4j()
            
            # Ensure vector index (for Browser and future vector queries)
            self.create_vector_index()
            
            # Compute similarity and create SIMILAR_TO edges (top-k)
            self.create_similarity_edges_topk()
            
            # Apply communities via targeted Leiden to reduce cluster count
            self.apply_leiden_target_communities(target=4)
            # Label nodes with community-specific labels for easy coloring in Browser
            self.label_nodes_by_community()
            
            # Analyze communities
            community_stats = self.analyze_communities()
            
            # Create visualizations
            graph_fig = self.visualize_graph('candidate_knowledge_graph_gds.html')
            
            print("\nAnalysis complete!")
            print("Files generated:")
            print("- candidate_knowledge_graph_gds.html (interactive graph)")
            print("- community_similarity_heatmap_gds.png (similarity heatmap)")
            print("\nTo view the graph:")
            print("1. Open candidate_knowledge_graph_gds.html in your browser")
            print("2. Or visit Neo4j Browser at http://localhost:7474")
            print("3. Use the Cypher queries provided in the README")
            
            return {
                'communities': self.communities,
                'community_stats': community_stats,
                'similarity_matrix': self.similarity_matrix,
                'embeddings': self.embeddings
            }
            
        finally:
            # Clean up GDS projections
            if self.driver:
                with self.driver.session() as session:
                    try:
                        # Check and drop 'candidates' projection
                        result = session.run("CALL gds.graph.exists('candidates')")
                        exists = result.single()['exists']
                        if exists:
                            session.run("CALL gds.graph.drop('candidates')")
                            print("Cleaned up 'candidates' graph projection")
                    except Exception as e:
                        print(f"Note: Could not clean up 'candidates' projection: {e}")
                    
                    try:
                        # Check and drop 'candidates_with_relationships' projection
                        result = session.run("CALL gds.graph.exists('candidates_with_relationships')")
                        exists = result.single()['exists']
                        if exists:
                            session.run("CALL gds.graph.drop('candidates_with_relationships')")
                            print("Cleaned up 'candidates_with_relationships' graph projection")
                    except Exception as e:
                        print(f"Note: Could not clean up 'candidates_with_relationships' projection: {e}")
                
                # Close connection
                self.close_neo4j()

def main():
    """Main function to run the analysis."""
    kg = CandidateKnowledgeGraph()
    results = kg.run_full_analysis()
    return results

if __name__ == "__main__":
    results = main()