#!/usr/bin/env python3
"""
Test script for the Candidate Knowledge Graph implementation with Neo4j and GDS.
This script demonstrates how to use the knowledge graph with tensor embeddings and GDS Leiden clustering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import CandidateKnowledgeGraph

def main():
    """Test the knowledge graph implementation with Neo4j GDS."""
    print("Testing Candidate Knowledge Graph with Neo4j GDS and Tensor Embeddings")
    print("=" * 80)
    
    # Path to the CSV file
    csv_path = '/Users/kateseeman/gitrepos/lovable-tool/src/data/candidates1-clay.csv'
    
    try:
        # Create the knowledge graph
        kg = CandidateKnowledgeGraph(csv_path)
        
        # Run the full analysis
        results = kg.run_full_analysis()
        
        if results is None:
            print("Analysis failed. Please check Neo4j connection and setup.")
            return None
        
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Print summary statistics
        print(f"Total candidates analyzed: {len(kg.df)}")
        print(f"Number of communities found: {len(set(kg.communities))}")
        print(f"Tensor embedding dimensions: {kg.tensor_embeddings.shape}")
        
        # Print community details
        print("\nCommunity Details:")
        for i, stats in enumerate(results['community_stats']):
            print(f"\nCommunity {i}:")
            print(f"  Size: {stats['size']} members")
            print(f"  Members: {', '.join(stats['members'])}")
            if stats['most_common_tech']:
                print(f"  Top Technologies: {', '.join([f'{tech}({count})' for tech, count in stats['most_common_tech'][:3]])}")
            if stats['most_common_affiliations']:
                print(f"  Top Affiliations: {', '.join([f'{aff}({count})' for aff, count in stats['most_common_affiliations'][:3]])}")
            print(f"  Average Experience: {stats['avg_experience']:.1f} years")
        
        print("\n" + "=" * 80)
        print("Files generated:")
        print("- candidate_knowledge_graph_gds.html (interactive visualization)")
        print("- community_similarity_heatmap_gds.png (similarity analysis)")
        print("\nNeo4j Browser available at: http://localhost:7474")
        print("Username: neo4j, Password: password")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nMake sure you have:")
        print("1. Installed all required dependencies: pip install -r requirements.txt")
        print("2. Started Neo4j with GDS plugin: python setup_neo4j.py")
        print("3. Neo4j is running and accessible at bolt://localhost:7687")
        return None

if __name__ == "__main__":
    results = main()
