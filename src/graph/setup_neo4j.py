#!/usr/bin/env python3
"""
Setup script for Neo4j with Graph Data Science plugin.
This script helps set up the environment for the candidate knowledge graph analysis.
"""

import os
import subprocess
import time
import requests
from neo4j import GraphDatabase

def create_env_file():
    """Create .env file with Neo4j configuration."""
    env_content = """# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional: Other configurations
# NEO4J_DATABASE=neo4j
"""
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_content)
        print(f"Created .env file at {env_path}")
    else:
        print(f".env file already exists at {env_path}")

def start_neo4j_docker():
    """Start Neo4j using Docker Compose."""
    print("Starting Neo4j with Docker Compose...")
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    docker_compose_path = os.path.join(project_root, 'docker-compose-neo4j.yml')
    
    try:
        # Start Neo4j
        subprocess.run([
            'docker-compose', '-f', docker_compose_path, 'up', '-d'
        ], check=True, cwd=project_root)
        
        print("Neo4j container started successfully!")
        print("Waiting for Neo4j to be ready...")
        
        # Wait for Neo4j to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get('http://localhost:7474', timeout=5)
                if response.status_code == 200:
                    print("Neo4j is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
            print(f"Waiting for Neo4j... (attempt {attempt + 1}/{max_attempts})")
        
        print("Neo4j failed to start within the expected time")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Neo4j: {e}")
        return False

def test_neo4j_connection():
    """Test connection to Neo4j."""
    print("Testing Neo4j connection...")
    
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            
            if test_value == 1:
                print("✅ Neo4j connection successful!")
                
                # Test GDS plugin
                try:
                    result = session.run("CALL gds.version()")
                    version = result.single()['gdsVersion']
                    print(f"✅ GDS plugin loaded successfully! Version: {version}")
                    return True
                except Exception as e:
                    print(f"❌ GDS plugin not available: {e}")
                    print("Please ensure the Graph Data Science plugin is installed")
                    return False
            else:
                print("❌ Neo4j connection test failed")
                return False
                
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

def check_existing_neo4j():
    """Check if Neo4j is already running."""
    print("Checking for existing Neo4j installation...")
    
    try:
        response = requests.get('http://localhost:7474', timeout=5)
        if response.status_code == 200:
            print("✅ Neo4j is already running at http://localhost:7474")
            return True
    except requests.exceptions.RequestException:
        pass
    
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()['test']
            if test_value == 1:
                print("✅ Neo4j is already running and accessible")
                driver.close()
                return True
    except Exception:
        pass
    
    return False

def provide_manual_setup_instructions():
    """Provide instructions for manual Neo4j setup."""
    print("\n" + "=" * 60)
    print("MANUAL NEO4J SETUP INSTRUCTIONS")
    print("=" * 60)
    print("\nSince Docker is not available, here are alternative ways to set up Neo4j:")
    print("\n1. NEO4J DESKTOP (Recommended for Mac/Windows):")
    print("   - Download from: https://neo4j.com/download/")
    print("   - Install Neo4j Desktop")
    print("   - Create a new project and database")
    print("   - Install the Graph Data Science plugin")
    print("   - Start the database")
    print("   - Set password to 'password'")
    
    print("\n2. NEO4J COMMUNITY SERVER:")
    print("   - Download from: https://neo4j.com/download/")
    print("   - Extract and run: ./bin/neo4j console")
    print("   - Install GDS plugin manually")
    print("   - Set password to 'password'")
    
    print("\n3. NEO4J AURA (Cloud):")
    print("   - Sign up at: https://console.neo4j.io/")
    print("   - Create a free instance")
    print("   - Update .env file with your connection details")
    
    print("\n4. USE EXISTING NEO4J:")
    print("   - If you have Neo4j running elsewhere")
    print("   - Update the .env file with your connection details")
    
    print("\nAfter setting up Neo4j:")
    print("1. Make sure GDS plugin is installed")
    print("2. Run: python setup_neo4j.py --test-only")
    print("3. If successful, run: python test_graph.py")
    print("=" * 60)

def main():
    """Main setup function."""
    print("Setting up Neo4j for Candidate Knowledge Graph Analysis")
    print("=" * 60)
    
    # Create .env file
    create_env_file()
    
    # Check if Neo4j is already running
    if check_existing_neo4j():
        if test_neo4j_connection():
            print("\n🎉 Neo4j is ready!")
            print("\nYou can now run the knowledge graph analysis:")
            print("cd src/graph")
            print("python test_graph.py")
            return True
        else:
            print("Neo4j is running but GDS plugin may not be available")
    
    # Check if Docker is available
    docker_available = False
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        print("✅ Docker is available")
        docker_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available")
    
    if docker_available:
        # Check if docker-compose is available
        try:
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
            print("✅ Docker Compose is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker Compose is not available")
            docker_available = False
    
    if docker_available:
        # Start Neo4j with Docker
        if start_neo4j_docker():
            if test_neo4j_connection():
                print("\n🎉 Neo4j setup completed successfully!")
                print("\nYou can now run the knowledge graph analysis:")
                print("cd src/graph")
                print("python test_graph.py")
                print("\nNeo4j Browser is available at: http://localhost:7474")
                print("Username: neo4j")
                print("Password: password")
                return True
            else:
                print("Neo4j started but GDS plugin may not be available")
        else:
            print("Failed to start Neo4j with Docker")
    else:
        provide_manual_setup_instructions()
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
