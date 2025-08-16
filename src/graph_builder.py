"""
Graph builder module for the Graph RAG Document Analyzer.

This module handles Neo4j graph database connections and knowledge graph construction.
"""

import os
from typing import List, Optional, Dict, Any
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from dotenv import load_dotenv


class GraphDatabaseConnection:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize the graph database connection.
        
        Args:
            uri: Neo4j database URI
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"Successfully connected to Neo4j at {self.uri}")
        except neo4j_exceptions.ServiceUnavailable:
            raise ConnectionError(f"Could not connect to Neo4j at {self.uri}")
        except neo4j_exceptions.AuthError:
            raise ConnectionError("Invalid Neo4j credentials")
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the current database schema."""
        with self.driver.session() as session:
            # Get node labels
            node_labels = session.run("CALL db.labels()").values()
            
            # Get relationship types
            relationships = session.run("CALL db.relationshipTypes()").values()
            
            # Get property keys
            properties = session.run("CALL db.propertyKeys()").values()
            
            return {
                "node_labels": [label[0] for label in node_labels],
                "relationships": [rel[0] for rel in relationships],
                "properties": [prop[0] for prop in properties]
            }


class GraphBuilder:
    """Builds knowledge graphs from document chunks using LLM extraction."""
    
    def __init__(
        self, 
        graph_connection: GraphDatabaseConnection,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Initialize the graph builder.
        
        Args:
            graph_connection: Neo4j database connection
            llm_model: OpenAI model for extraction
            temperature: LLM temperature for consistent extraction
        """
        self.graph_connection = graph_connection
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)
    
    def build_graph_from_document(self, chunks: List[Document]) -> Dict[str, int]:
        """
        Build a knowledge graph from document chunks.
        
        Args:
            chunks: List of document chunks to process
            
        Returns:
            Dictionary with statistics about the created graph
        """
        total_nodes = 0
        total_relationships = 0
        
        print(f"Processing {len(chunks)} document chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Convert chunk to graph documents
                graph_documents = self.graph_transformer.convert_to_graph_documents([chunk])
                
                # Add to Neo4j database
                for graph_doc in graph_documents:
                    nodes_added, rels_added = self._add_graph_document_to_db(graph_doc)
                    total_nodes += nodes_added
                    total_relationships += rels_added
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        return {
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "chunks_processed": len(chunks)
        }
    
    def _add_graph_document_to_db(self, graph_document) -> tuple[int, int]:
        """
        Add a graph document to the Neo4j database.
        
        Args:
            graph_document: LangChain graph document
            
        Returns:
            Tuple of (nodes_added, relationships_added)
        """
        nodes_added = 0
        relationships_added = 0
        
        with self.graph_connection.driver.session() as session:
            # Add nodes
            for node in graph_document.nodes:
                result = session.run(
                    "MERGE (n {id: $id}) "
                    "SET n += $properties "
                    "RETURN count(n) as nodes_created",
                    id=node.id,
                    properties=dict(node.properties) if node.properties else {}
                )
                nodes_added += result.single()["nodes_created"]
            
            # Add relationships
            for relationship in graph_document.relationships:
                # Create a safe relationship type name (remove spaces, special chars)
                safe_rel_type = "".join(c for c in relationship.type if c.isalnum() or c == "_").upper()
                if not safe_rel_type:
                    safe_rel_type = "RELATED_TO"
                
                result = session.run(f"""
                    MATCH (source {{id: $source_id}}) 
                    MATCH (target {{id: $target_id}}) 
                    MERGE (source)-[r:{safe_rel_type}]->(target) 
                    SET r += $properties 
                    RETURN count(r) as rels_created
                """,
                    source_id=relationship.source.id,
                    target_id=relationship.target.id,
                    properties=dict(relationship.properties) if relationship.properties else {}
                )
                relationships_added += result.single()["rels_created"]
        
        return nodes_added, relationships_added


def get_graph_connection() -> GraphDatabaseConnection:
    """
    Create and return a graph database connection using environment variables.
    
    Returns:
        Configured GraphDatabaseConnection instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()
    
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, username, password]):
        missing = [var for var, val in [
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username), 
            ("NEO4J_PASSWORD", password)
        ] if not val]
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    connection = GraphDatabaseConnection(uri, username, password)
    connection.connect()
    return connection