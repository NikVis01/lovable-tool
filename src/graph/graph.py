import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

# --------------------
# CONFIG
# --------------------
CSV_PATH = "data/candidates.csv"
TEXT_COLS = ["Name","LinkedIn","Github","Affiliations","Tech stack","Location","Most Notable Company","Coolest Problem","Degree"]
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
    def _ensure_canonical_columns(self):
        """Map variant CSV headers to canonical names and backfill missing ones."""
        def norm(s: str) -> str:
            return "".join(ch for ch in s.lower().strip() if ch.isalnum())

        colmap = {norm(c): c for c in self.df.columns}
        aliases = {
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
                    id: $id,
                    name: $name,
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
                    'id': int(row[ID_COL]),
                    'name': row.get('Name', ''),
                    'linkedin': row.get('LinkedIn', ''),
                    'github': row.get('Github', ''),
                    'tech_stack': row.get('Tech stack', ''),
                    'affiliations': row.get('Affiliations', ''),
                    'location': row.get('Location', ''),
                    'years_experience': row.get('Years Experience', 0.0),
                    'company': row.get('Most Notable Company', ''),
                    'problem': row.get('Coolest Problem', ''),
                    'degree': row.get('Degree', ''),
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
    
    def compute_similarity_with_gds(self, threshold=0.6):
        """Use GDS to compute similarity and create relationships."""
        print("Computing similarity using GDS...")
        
        
        # Use the embeddings we already created instead of GDS FastRP
        # This avoids the session issue with GDS projections
        print("Using pre-computed embeddings for similarity analysis...")
        
        # Compute cosine similarity directly from our embeddings
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        # Create relationships based on similarity threshold
        relationships_created = 0
        with self.driver.session() as session:
            for i in range(len(self.df)):
                for j in range(i + 1, len(self.df)):
                    similarity = self.similarity_matrix[i, j]
                    if similarity > threshold:
                        session.run("""
                            MATCH (a:Candidate), (b:Candidate)
                            WHERE a.id = $id1 AND b.id = $id2
                            CREATE (a)-[r:SIMILAR_TO {
                                similarity: $similarity,
                                threshold: $threshold
                            }]->(b)
                        """, {
                            'id1': i,
                            'id2': j,
                            'similarity': float(similarity),
                            'threshold': threshold
                        })
                        relationships_created += 1
            
            print(f"Created {relationships_created} similarity relationships")
            return self.similarity_matrix

    def apply_cosine_communities(self, threshold: float | None = None):
        """Compute communities using pure cosine-similarity connectivity (no GDS).

        Strategy: Build an implicit graph where edges exist for pairs with
        similarity > threshold, then assign communities as connected components.
        """
        print("Assigning communities by cosine similarity (no GDS)...")
        if threshold is None:
            try:
                threshold = float(os.getenv("COSINE_SIM_THRESHOLD", "0.25"))
            except ValueError:
                threshold = 0.25
        if self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.embeddings)

        n = len(self.df)
        communities = [-1] * n

        def dfs(start: int, cid: int):
            stack = [start]
            communities[start] = cid
            while stack:
                u = stack.pop()
                # Iterate neighbors by threshold
                for v in range(n):
                    if v == u:
                        continue
                    if communities[v] != -1:
                        continue
                    if self.similarity_matrix[u, v] > threshold:
                        communities[v] = cid
                        stack.append(v)

        cid = 0
        for i in range(n):
            if communities[i] == -1:
                dfs(i, cid)
                cid += 1

        # Persist community to Neo4j
        with self.driver.session() as session:
            for idx, comm in enumerate(communities):
                session.run(
                    "MATCH (c:Candidate) WHERE c.id = $id SET c.community = $community",
                    { 'id': idx, 'community': int(comm) }
                )

        self.communities = communities
        # Mirror to dataframe for downstream analysis
        self.df['community'] = self.communities
        print(f"Assigned {cid} cosine communities")
        return communities

    def label_nodes_by_community(self, max_old_labels: int = 100):
        """Attach label Community{n} to each Candidate node based on c.community.

        Also attempts to remove a bounded set of old Community labels to avoid label buildup.
        """
        print("Updating node labels by community for Neo4j Browser coloring...")
        with self.driver.session() as session:
            # Best-effort cleanup of prior community labels (0..max_old_labels-1)
            for k in range(max_old_labels):
                lbl = f"Community{k}"
                try:
                    session.run(f"MATCH (c:Candidate:{lbl}) REMOVE c:{lbl}")
                except Exception:
                    pass
            # Apply current labels
            for idx, comm in enumerate(self.communities):
                lbl = f"Community{int(comm)}"
                session.run(
                    f"MATCH (c:Candidate) WHERE c.id = $id SET c:{lbl}",
                    { 'id': idx }
                )

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

        # Persist results
        self.communities = communities
        self.df['community'] = self.communities
        with self.driver.session() as session:
            for idx, comm in enumerate(self.communities):
                session.run(
                    "MATCH (c:Candidate) WHERE c.id = $id SET c.community = $comm",
                    { 'id': idx, 'comm': int(comm) }
                )
        print(f"Assigned {cid} mutual-kNN communities (k={k})")
        return communities
    
    def apply_leiden_clustering_gds(self, random_seed: int = 23, include_intermediate: bool = False,
                                    min_communities: int = 3, start_resolution: float = 1.0,
                                    max_resolution: float = 64.0, resolution_multiplier: float = 1.5):
        """Apply Leiden clustering using GDS and try to reach at least min_communities.

        We tune the 'resolution' parameter until the number of discovered communities
        is >= min_communities or we reach max_resolution. See docs: https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/
        """
        print("Applying Leiden clustering using GDS...")
        
        with self.driver.session() as session:
            # Create a new graph projection with relationships
            try:
                result = session.run("CALL gds.graph.exists('candidates_with_relationships')")
                exists = result.single()['exists']
                if exists:
                    session.run("CALL gds.graph.drop('candidates_with_relationships')")
                    print("Dropped existing 'candidates_with_relationships' graph projection")
            except Exception as e:
                print(f"Note: Could not check/drop existing graph projection: {e}")
            
            # Create the graph projection in the same session (only numeric properties)
            # Make relationships undirected for Leiden algorithm
            session.run("""
                CALL gds.graph.project(
                    'candidates_with_relationships',
                    {
                        Candidate: {
                            properties: ['years_experience']
                        }
                    },
                    {
                        SIMILAR_TO: {
                            properties: ['similarity'],
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
            """)
            print("Created 'candidates_with_relationships' graph projection with undirected relationships")
            
            # Sweep resolution to target a minimum number of communities
            resolution = float(start_resolution)
            communities = None
            while True:
                cfg = {
                    'relationshipWeightProperty': 'similarity',
                    'randomSeed': random_seed,
                    'resolution': resolution,
                }
                if include_intermediate:
                    cfg['includeIntermediateCommunities'] = True
                result = session.run(
                    "CALL gds.leiden.stream('candidates_with_relationships', $cfg) "
                    "YIELD nodeId, communityId RETURN nodeId, communityId",
                    { 'cfg': cfg }
                )
                tmp = {}
                for record in result:
                    tmp[record['nodeId']] = record['communityId']
                count = len(set(tmp.values()))
                print(f"Leiden resolution={resolution} -> {count} communities")
                communities = tmp
                if count >= min_communities or resolution >= max_resolution:
                    break
                resolution *= resolution_multiplier
            
            # Update candidates with community information
            for node_id, community_id in communities.items():
                session.run("""
                    MATCH (c:Candidate)
                    WHERE c.id = $node_id
                    SET c.community = $community_id
                """, {
                    'node_id': node_id,
                    'community_id': community_id
                })
            
            # Get community assignments for our dataframe
            self.communities = []
            for idx in range(len(self.df)):
                result = session.run("""
                    MATCH (c:Candidate)
                    WHERE c.id = $id
                    RETURN c.community as community
                """, {'id': idx})
                
                record = result.single()
                if record:
                    self.communities.append(record['community'])
                else:
                    self.communities.append(0)
            
            # Add community information to the dataframe
            self.df['community'] = self.communities
            
            print(f"Found {len(set(self.communities))} communities")
            for i, community in enumerate(set(self.communities)):
                members = [self.df.iloc[j]['Name'] for j in range(len(self.communities)) if self.communities[j] == community]
                print(f"Community {i}: {len(members)} members - {', '.join(members[:3])}{'...' if len(members) > 3 else ''}")
            
            return self.communities
    
    def get_graph_data_for_visualization(self):
        """Extract graph data from Neo4j for visualization."""
        print("Extracting graph data for visualization...")
        
        with self.driver.session() as session:
            # Get nodes and their properties
            nodes_result = session.run("""
                MATCH (c:Candidate)
                RETURN c.id as id, c.name as name, c.community as community,
                       c.tech_stack as tech_stack, c.affiliations as affiliations,
                       c.years_experience as experience
            """)
            
            nodes = []
            for record in nodes_result:
                nodes.append({
                    'id': record['id'],
                    'name': record['name'],
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
        
        # Color map for communities (unique mapping)
        color_map = self._build_community_color_map()
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            x = np.cos(angle)
            y = np.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            name = node['name']
            tech = node['tech_stack']
            community = node['community']
            
            node_text.append(f"{name}<br>Tech: {tech}<br>Community: {community}")
            node_comms.append(community)
            
            # Size based on number of connections
            connections = len([e for e in edges if e['source'] == node['id'] or e['target'] == node['id']])
            node_sizes.append(max(10, min(30, connections * 3)))
        
        # Build separate traces per community for distinct coloring and legend
        node_traces = []
        names = [node['name'] for node in nodes]
        unique_comms = sorted(set(node_comms))
        for comm in unique_comms:
            idxs = [i for i,c in enumerate(node_comms) if c == comm]
            node_traces.append(
                go.Scatter(
                    x=[node_x[i] for i in idxs],
                    y=[node_y[i] for i in idxs],
                    mode='markers+text',
                    name=f"Community {comm}",
                    hoverinfo='text',
                    text=[names[i] for i in idxs],
                    textposition="middle center",
                    hovertext=[node_text[i] for i in idxs],
                    marker=dict(
                        size=[node_sizes[i] for i in idxs],
                        color=color_map.get(comm, '#8888ff'),
                        line=dict(width=2, color='black')
                    )
                )
            )
        
        # Create the plot
        fig = go.Figure(data=[edge_trace] + node_traces,
                       layout=go.Layout(
                           title=dict(
                               text='Candidate Knowledge Graph with GDS Leiden Communities',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents number of connections. Color represents community.",
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
            
            # Compute similarity and create SIMILAR_TO edges
            self.compute_similarity_with_gds()
            
            # Apply communities via mutual k-NN to encourage multiple clusters
            self.apply_mutual_knn_communities()
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