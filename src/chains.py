"""
RAG chains module for the Graph RAG Document Analyzer.

This module implements the retrieval and generation chains for graph-based Q&A.
"""

import os
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


class GraphRAGChain:
    """Implements Graph RAG functionality using Neo4j and OpenAI."""
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        extraction_model: str = "gpt-4o-mini",
        generation_model: str = "gpt-4o",
        verbose: bool = True
    ):
        """
        Initialize the Graph RAG chain.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            extraction_model: Model for query generation
            generation_model: Model for final answer generation
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Initialize LLMs
        self.extraction_llm = ChatOpenAI(
            model=extraction_model,
            temperature=0.0
        )
        
        self.generation_llm = ChatOpenAI(
            model=generation_model,
            temperature=0.3
        )
        
        # Initialize generation chain
        self.generation_chain = self._create_generation_chain()
        
        # Get schema info
        self.schema_info = self._get_schema_info()
    
    def _create_generation_chain(self):
        """Create the final answer generation chain."""
        system_prompt = (
            "You are a document analysis assistant. Your job is to answer questions directly using the entities found in the document's knowledge graph.\n\n"
            
            "ANSWER FORMAT:\n"
            "- Give direct, factual answers based on the entities shown\n"
            "- Start with the main answer, then provide supporting details if relevant\n"
            "- Use simple, clear language\n"
            "- Avoid generic explanations not based on the document content\n\n"
            
            "EXAMPLE RESPONSES:\n"
            "Q: What is intellectual property?\n"
            "A: Based on the document, intellectual property refers to [specific definition/description from the entity details].\n\n"
            
            "Q: What are the AI policies?\n"  
            "A: The document mentions these AI policies: [list the specific policy entities found].\n\n"
            
            "CONTEXT FROM DOCUMENT:\n{context}\n\n"
            
            "If the context doesn't contain relevant entities for the question, say 'The document doesn't contain specific information about [topic]' rather than giving generic answers."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        return prompt | self.generation_llm | StrOutputParser()
    
    def _get_schema_info(self) -> str:
        """Get database schema without APOC procedures."""
        with self.driver.session() as session:
            # Get node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record[0] for record in labels_result]
            
            # Get relationship types  
            rels_result = session.run("CALL db.relationshipTypes()")
            relationships = [record[0] for record in rels_result]
            
            schema = f"""
            Node Labels: {', '.join(labels) if labels else 'None'}
            Relationship Types: {', '.join(relationships) if relationships else 'None'}
            """
            return schema.strip()
    
    def _generate_cypher_query(self, question: str) -> str:
        """Generate Cypher query from natural language question."""
        import re
        
        question_lower = question.lower()
        
        # Extract key terms, preserving multi-word concepts
        # First, look for quoted phrases or obvious compound terms
        compound_terms = re.findall(r'["\'](.*?)["\']', question_lower)
        
        # Extract individual meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
        
        # Remove common question words and function words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'are', 'the', 'and', 'for', 
            'with', 'about', 'does', 'did', 'can', 'will', 'would', 'should', 'could',
            'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'your'
        }
        key_terms = [word for word in words if word not in stop_words]
        
        # Combine compound terms and key terms
        all_terms = compound_terms + key_terms
        
        # For compound questions with "and", try to find entities that match both concepts
        if " and " in question_lower and len(all_terms) >= 2:
            # Build a query that looks for nodes containing any of the key terms
            # but prioritize exact matches and limit results more aggressively
            conditions = []
            for term in all_terms[:3]:  # Limit to top 3 terms
                conditions.append(f"toLower(n.id) CONTAINS toLower('{term}')")
            
            cypher_query = f"""
                MATCH (n) 
                WHERE {' OR '.join(conditions)}
                RETURN n 
                ORDER BY 
                    CASE 
                        WHEN toLower(n.id) = toLower('{all_terms[0]}') THEN 1
                        WHEN toLower(n.id) CONTAINS toLower('{all_terms[0]}') THEN 2
                        ELSE 3
                    END
                LIMIT 6
            """
        elif len(all_terms) >= 1:
            # Single concept query - look for best matches
            primary_term = all_terms[0]
            cypher_query = f"""
                MATCH (n) 
                WHERE toLower(n.id) CONTAINS toLower('{primary_term}')
                RETURN n 
                ORDER BY 
                    CASE 
                        WHEN toLower(n.id) = toLower('{primary_term}') THEN 1
                        WHEN toLower(n.id) CONTAINS toLower('{primary_term}') THEN 2
                        ELSE 3
                    END
                LIMIT 5
            """
        else:
            # Fallback - get some nodes
            cypher_query = "MATCH (n) RETURN n LIMIT 5"
            
        return cypher_query.strip()
    
    def _execute_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            if self.verbose:
                print(f"Cypher execution error: {e}")
            return []
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the knowledge graph with a natural language question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Step 1: Generate Cypher query
            generated_cypher = self._generate_cypher_query(question)
            if self.verbose:
                print(f"Generated Cypher: {generated_cypher}")
            
            # Step 2: Execute query to get context
            raw_context = self._execute_cypher_query(generated_cypher)
            
            # Step 3: Format context for generation
            formatted_context = self._format_context(raw_context)
            
            # Step 4: Generate final answer
            if formatted_context and formatted_context != "No relevant information found in the knowledge graph.":
                final_answer = self.generation_chain.invoke({
                    "context": formatted_context,
                    "question": question
                })
            else:
                final_answer = "I couldn't find relevant information in the knowledge graph to answer your question."
            
            return {
                "answer": final_answer,
                "generated_cypher": generated_cypher,
                "raw_context": raw_context,
                "formatted_context": formatted_context
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error in query processing: {str(e)}")
            
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "error": str(e)
            }
    
    def _format_context(self, raw_context: List[Dict[str, Any]]) -> str:
        """
        Format raw graph context for the generation model.
        
        Args:
            raw_context: Raw context from Cypher query
            
        Returns:
            Formatted context string
        """
        if not raw_context:
            return "No relevant information found in the knowledge graph."
        
        # Extract and deduplicate entity names
        entities = set()
        entity_details = {}
        
        for item in raw_context:
            if isinstance(item, dict):
                for key, value in item.items():
                    if value is not None:
                        # Handle node objects
                        if hasattr(value, 'get') and 'id' in value:
                            entity_id = value.get('id', 'Unknown')
                            entities.add(entity_id)
                            # Store the most relevant properties (exclude verbose ones)
                            relevant_props = {}
                            for prop_key, prop_val in value.items():
                                if (prop_key != 'id' and prop_val is not None and 
                                    len(str(prop_val)) < 200):  # Skip very long properties
                                    relevant_props[prop_key] = prop_val
                            if relevant_props:
                                entity_details[entity_id] = relevant_props
                                
                        elif isinstance(value, dict) and 'id' in value:
                            entity_id = value['id']
                            entities.add(entity_id)
                            # Store relevant properties
                            relevant_props = {}
                            for prop_key, prop_val in value.items():
                                if (prop_key != 'id' and prop_val is not None and 
                                    len(str(prop_val)) < 200):
                                    relevant_props[prop_key] = prop_val
                            if relevant_props:
                                entity_details[entity_id] = relevant_props
        
        if not entities:
            return "No relevant information found in the knowledge graph."
        
        # Format entities in a clean, concise way
        formatted_lines = []
        for entity in sorted(entities):
            if entity in entity_details and entity_details[entity]:
                # Only show the most important property if any exist
                props = entity_details[entity]
                if props:
                    # Show just the first meaningful property
                    prop_key, prop_val = next(iter(props.items()))
                    formatted_lines.append(f"• {entity}: {prop_val}")
                else:
                    formatted_lines.append(f"• {entity}")
            else:
                formatted_lines.append(f"• {entity}")
        
        return "\n".join(formatted_lines)
    
    def get_graph_schema(self) -> str:
        """Get the current graph schema."""
        return self.schema_info
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()


def create_graph_rag_chain() -> GraphRAGChain:
    """
    Create a Graph RAG chain using environment variables.
    
    Returns:
        Configured GraphRAGChain instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()
    
    # Required environment variables
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Optional model configuration
    extraction_model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")
    generation_model = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        missing = [var for var, val in [
            ("NEO4J_URI", neo4j_uri),
            ("NEO4J_USERNAME", neo4j_username),
            ("NEO4J_PASSWORD", neo4j_password)
        ] if not val]
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    return GraphRAGChain(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        extraction_model=extraction_model,
        generation_model=generation_model
    )


def run_graph_rag(question: str) -> str:
    """
    Convenience function to run a Graph RAG query.
    
    Args:
        question: User's question
        
    Returns:
        Generated answer
    """
    chain = create_graph_rag_chain()
    result = chain.query(question)
    return result["answer"]