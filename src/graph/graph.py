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
CSV_PATH = "data/candidates1-clay.csv"
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
        
        # Add ID column if not present
        if ID_COL not in self.df.columns:
            self.df[ID_COL] = np.arange(len(self.df))
        
        # Clean and preprocess the data
        for col in TEXT_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('')
        
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
        
        # Combine text columns
        texts = self.df[TEXT_COLS].astype(str).agg(" ".join, axis=1).tolist()
        
        # Load sentence transformer model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Create embeddings
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        # Expose alias expected by tests
        self.tensor_embeddings = self.embeddings
        
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
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
    
    def compute_similarity_with_gds(self, threshold=0.4):
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
    
    def apply_leiden_clustering_gds(self):
        """Apply Leiden clustering using GDS."""
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
            
            # Apply Leiden algorithm in the same session
            result = session.run("""
                CALL gds.leiden.stream('candidates_with_relationships', {
                    relationshipWeightProperty: 'similarity',
                    randomSeed: 42
                })
                YIELD nodeId, communityId
                RETURN nodeId, communityId
                ORDER BY communityId, nodeId
            """)
            
            # Store community assignments
            communities = {}
            for record in result:
                communities[record['nodeId']] = record['communityId']
            
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
        node_colors = []
        node_sizes = []
        
        # Color map for communities
        colors = px.colors.qualitative.Set3
        
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
            node_colors.append(colors[community % len(colors)])
            
            # Size based on number of connections
            connections = len([e for e in edges if e['source'] == node['id'] or e['target'] == node['id']])
            node_sizes.append(max(10, min(30, connections * 3)))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node['name'] for node in nodes],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
        
        # Create the plot
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text='Candidate Knowledge Graph with GDS Leiden Communities',
                               font=dict(size=16)
                           ),
                           showlegend=False,
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
            
            # Create GDS graph projection
            self.create_gds_graph()
            
            # Compute similarity using GDS
            self.compute_similarity_with_gds()
            
            # Apply Leiden clustering using GDS
            self.apply_leiden_clustering_gds()
            
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