#!/usr/bin/env python3
"""
Test script for the Neo4j-based Candidate Knowledge Graph implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import CandidateKnowledgeGraph

def main():
    """Test the Neo4j knowledge graph implementation."""
    print("Testing Candidate Knowledge Graph with Neo4j GDS")
    print("=" * 60)
    
    try:
        # Create the knowledge graph
        kg = CandidateKnowledgeGraph()
        
        # Run the full analysis
        results = kg.run_full_analysis()
        
        if results is None:
            print("Analysis failed. Please check Neo4j connection and setup.")
            return None
        
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Print summary statistics
        print(f"Total candidates analyzed: {len(kg.df)}")
        print(f"Number of communities found: {len(set(kg.communities))}")
        print(f"Embedding dimensions: {kg.embeddings.shape}")
        
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
        
        print("\n" + "=" * 60)
        print("Files generated:")
        print("- candidate_knowledge_graph_gds.html (interactive visualization)")
        print("- community_similarity_heatmap_gds.png (similarity analysis)")
        print("\nNeo4j Browser available at: http://localhost:7474")
        print("Username: neo4j, Password: Password")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nMake sure you have:")
        print("1. Neo4j running with GDS plugin")
        print("2. Correct connection details in graph.py")
        print("3. All required dependencies installed")
        return None

if __name__ == "__main__":
    results = main()
