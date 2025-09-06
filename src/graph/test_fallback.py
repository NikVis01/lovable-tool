#!/usr/bin/env python3
"""
Test script for the Candidate Knowledge Graph fallback implementation.
This script works without Neo4j and uses a simplified clustering approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_fallback import CandidateKnowledgeGraphFallback

def main():
    """Test the fallback knowledge graph implementation."""
    print("Testing Candidate Knowledge Graph (Fallback Mode - No Neo4j Required)")
    print("=" * 80)
    
    # Path to the CSV file
    csv_path = '/Users/kateseeman/gitrepos/lovable-tool/src/data/candidates1-clay.csv'
    
    try:
        # Create the knowledge graph
        kg = CandidateKnowledgeGraphFallback(csv_path)
        
        # Run the full analysis
        results = kg.run_full_analysis()
        
        if results is None:
            print("Analysis failed.")
            return None
        
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Print summary statistics
        print(f"Total candidates analyzed: {len(kg.df)}")
        print(f"Number of communities found: {len(set(kg.communities))}")
        print(f"Tensor embedding dimensions: {kg.tensor_embeddings.shape}")
        print(f"Number of similarity relationships: {len(kg.graph_edges)}")
        
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
        print("- candidate_knowledge_graph_fallback.html (interactive visualization)")
        print("- community_similarity_heatmap_fallback.png (similarity analysis)")
        print("\nNote: This is a simplified version without Neo4j GDS.")
        print("For full functionality, set up Neo4j and use the main implementation.")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nMake sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")
        return None

if __name__ == "__main__":
    results = main()
