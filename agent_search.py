import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable, Any
import chromadb
from chromadb.config import Settings
import hashlib
import frontmatter
import nltk
from nltk.tokenize import sent_tokenize
import time
import logging
from dataclasses import dataclass, field
from enum import Enum


# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    message_type: str  # 'query', 'result', 'request_help', 'share_context'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class ResearchAgent:
    def __init__(self, name: str, role: str, specialty: str, search_system: 'MarkdownSearchSystem'):
        self.name = name
        self.role = role
        self.specialty = specialty
        self.search_system = search_system
        self.history = []
        self.context = {}
        self.discovered_topics = set()
        self.search_refinements = []
        self.collaboration_messages = []
        
    def process_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Process a query using the agent's specialty and context."""
        logger.info(f"{self.name} processing query: {query[:100]}...")
        
        # Refine query based on agent's specialty
        refined_query = self._refine_query_by_specialty(query, context)
        
        # Search with multiple strategies
        search_results = self._multi_strategy_search(refined_query)
        
        # Analyze results
        analysis = self._analyze_results(search_results, query)
        
        # Update agent's context and knowledge
        self._update_context(analysis)
        
        # Prepare response
        response = {
            'agent': self.name,
            'original_query': query,
            'refined_query': refined_query,
            'search_results': search_results,
            'analysis': analysis,
            'discovered_topics': list(self.discovered_topics),
            'confidence': analysis.get('confidence', 0.5),
            'suggestions': analysis.get('suggestions', []),
            'needs_collaboration': analysis.get('needs_collaboration', False)
        }
        
        self.history.append(response)
        return response
    
    def _refine_query_by_specialty(self, query: str, context: Dict = None) -> str:
        """Refine query based on agent's specialty."""
        context = context or {}
        
        specialty_prompts = {
            'contextualizer': f"Find background information, definitions, and context for: {query}",
            'synthesizer': f"Summarize and synthesize information about: {query}",
            'validator': f"Find evidence and validation for claims about: {query}",
            'explorer': f"Discover related topics and connections to: {query}",
            'temporal': f"Find chronological information and timelines about: {query}",
            'technical': f"Find technical details, specifications, and implementations of: {query}"
        }
        
        base_refinement = specialty_prompts.get(
            self.specialty.lower(), 
            f"Find detailed information about: {query}"
        )
        
        # Add context from previous searches
        if self.context:
            context_terms = ', '.join(list(self.discovered_topics)[:5])
            if context_terms:
                base_refinement += f" Related context: {context_terms}"
        
        return base_refinement
    
    def _multi_strategy_search(self, query: str) -> List[SearchResult]:
        """Perform multiple search strategies and combine results."""
        all_results = []
        
        # Strategy 1: Direct semantic search
        direct_results = self.search_system.search_documents(query, n_results=8)
        for result in direct_results:
            all_results.append(SearchResult(
                content=result['content'],
                metadata=result['metadata'],
                relevance_score=result['relevance_score'],
                source='direct_search'
            ))
        
        # Strategy 2: Keyword extraction and search
        keywords = self._extract_keywords(query)
        for keyword in keywords[:3]:  # Top 3 keywords
            keyword_results = self.search_system.search_documents(keyword, n_results=3)
            for result in keyword_results:
                all_results.append(SearchResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    relevance_score=result['relevance_score'] * 0.8,  # Slightly lower weight
                    source='keyword_search'
                ))
        
        # Strategy 3: Search based on discovered topics
        for topic in list(self.discovered_topics)[:2]:
            topic_results = self.search_system.search_documents(f"{query} {topic}", n_results=2)
            for result in topic_results:
                all_results.append(SearchResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    relevance_score=result['relevance_score'] * 0.9,
                    source='topic_expansion'
                ))
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        return sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)[:10]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        keywords = [word for word in words if word not in stop_words]
        return list(set(keywords))[:5]  # Return top 5 unique keywords
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results based on content similarity."""
        unique_results = []
        seen_hashes = set()
        
        for result in results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                unique_results.append(result)
                seen_hashes.add(content_hash)
        
        return unique_results
    
    def _analyze_results(self, results: List[SearchResult], original_query: str) -> Dict[str, Any]:
        """Analyze search results and extract insights."""
        if not results:
            return {
                'confidence': 0.0,
                'insights': [],
                'suggestions': ['Try rephrasing the query', 'Search for related terms'],
                'needs_collaboration': True
            }
        
        # Extract topics and themes
        all_content = ' '.join([r.content for r in results])
        topics = self._extract_keywords(all_content)
        self.discovered_topics.update(topics)
        
        # Calculate confidence based on result quality
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        confidence = min(avg_relevance * 1.2, 1.0)  # Boost confidence slightly
        
        # Generate insights based on agent specialty
        insights = self._generate_specialty_insights(results, original_query)
        
        # Determine if collaboration is needed
        needs_collaboration = confidence < 0.6 or len(results) < 3
        
        suggestions = []
        if confidence < 0.5:
            suggestions.append("Consider refining the search terms")
        if len(set(r.metadata.get('filename', '') for r in results)) < 2:
            suggestions.append("Search across more document sources")
        
        return {
            'confidence': confidence,
            'insights': insights,
            'discovered_topics': topics,
            'suggestions': suggestions,
            'needs_collaboration': needs_collaboration,
            'result_count': len(results),
            'avg_relevance': avg_relevance
        }
    
    def _generate_specialty_insights(self, results: List[SearchResult], query: str) -> List[str]:
        """Generate insights based on the agent's specialty."""
        insights = []
        
        if self.specialty.lower() == 'contextualizer':
            dates = [r.metadata.get('date', '') for r in results if r.metadata.get('date')]
            if dates:
                insights.append(f"Found information spanning dates: {min(dates)} to {max(dates)}")
            
            sources = set(r.metadata.get('filename', '') for r in results)
            insights.append(f"Information found across {len(sources)} document sources")
            
        elif self.specialty.lower() == 'synthesizer':
            word_count = sum(len(r.content.split()) for r in results)
            insights.append(f"Synthesized {word_count} words of content")
            
            themes = self._extract_keywords(' '.join(r.content for r in results))
            if themes:
                insights.append(f"Key themes identified: {', '.join(themes[:5])}")
        
        elif self.specialty.lower() == 'validator':
            high_confidence_results = [r for r in results if r.relevance_score > 0.7]
            insights.append(f"{len(high_confidence_results)}/{len(results)} results have high confidence")
            
        elif self.specialty.lower() == 'explorer':
            related_topics = list(self.discovered_topics)[-10:]  # Recent discoveries
            if related_topics:
                insights.append(f"Discovered related topics: {', '.join(related_topics[:5])}")
        
        return insights
    
    def _update_context(self, analysis: Dict[str, Any]):
        """Update agent's internal context based on analysis."""
        self.context.update({
            'last_query_confidence': analysis.get('confidence', 0.0),
            'recent_topics': list(self.discovered_topics)[-20:],  # Keep recent topics
            'last_search_timestamp': datetime.now().isoformat()
        })
    
    def collaborate_with(self, other_agent: 'ResearchAgent', message: AgentMessage) -> Optional[AgentMessage]:
        """Handle collaboration with another agent."""
        logger.info(f"{self.name} collaborating with {other_agent.name}")
        
        if message.message_type == 'request_help':
            # Provide help based on our specialty
            query = message.content
            our_results = self.process_query(query, message.metadata)
            
            response = AgentMessage(
                sender=self.name,
                recipient=other_agent.name,
                content=f"Here's what I found: {our_results['analysis']['insights']}",
                message_type='share_context',
                metadata=our_results
            )
            return response
        
        elif message.message_type == 'share_context':
            # Incorporate shared context
            shared_topics = message.metadata.get('discovered_topics', [])
            self.discovered_topics.update(shared_topics)
            return None
        
        return None

class ResearchOrchestrator:
    def __init__(self, search_system: 'MarkdownSearchSystem', goal: str, max_iterations: int = 5):
        self.search_system = search_system
        self.goal = goal
        self.max_iterations = max_iterations
        self.agents = self._create_agents()
        self.iteration_count = 0
        self.final_report = ""
        self.status = TaskStatus.PENDING
        self.conversation_log = []
        
    def _create_agents(self) -> List[ResearchAgent]:
        """Create specialized research agents."""
        agent_configs = [
            ("ContextBot", "Information Contextualizer", "contextualizer"),
            ("SynthAI", "Content Synthesizer", "synthesizer"),
            ("ValidatorPro", "Information Validator", "validator"),
            ("ExplorerX", "Topic Explorer", "explorer"),
            ("ChronoAgent", "Temporal Analyzer", "temporal"),
            ("TechSpec", "Technical Specialist", "technical")
        ]
        
        agents = []
        for name, role, specialty in agent_configs:
            agent = ResearchAgent(name, role, specialty, self.search_system)
            agents.append(agent)
            
        return agents
    
    def run(self) -> Dict[str, Any]:
        """Execute the multi-agent research task."""
        logger.info(f"Starting multi-agent research: {self.goal}")
        self.status = TaskStatus.IN_PROGRESS
        
        try:
            # Phase 1: Initial exploration by all agents
            initial_results = self._initial_exploration()
            
            # Phase 2: Collaborative refinement
            refined_results = self._collaborative_refinement(initial_results)
            
            # Phase 3: Synthesis and validation
            final_synthesis = self._final_synthesis(refined_results)
            
            # Phase 4: Generate final report
            self.final_report = self._generate_final_report(final_synthesis)
            
            self.status = TaskStatus.COMPLETED
            
            return {
                'status': self.status.value,
                'goal': self.goal,
                'iterations': self.iteration_count,
                'final_report': self.final_report,
                'agent_contributions': {agent.name: len(agent.history) for agent in self.agents},
                'total_discovered_topics': len(set().union(*[agent.discovered_topics for agent in self.agents])),
                'conversation_log': self.conversation_log
            }
            
        except Exception as e:
            logger.error(f"Research task failed: {e}")
            self.status = TaskStatus.FAILED
            return {
                'status': self.status.value,
                'error': str(e),
                'partial_results': self.final_report
            }
    
    def _initial_exploration(self) -> Dict[str, Any]:
        """Phase 1: Each agent explores the goal independently."""
        logger.info("Phase 1: Initial exploration")
        exploration_results = {}
        
        for agent in self.agents:
            result = agent.process_query(self.goal)
            exploration_results[agent.name] = result
            
            log_entry = f"{agent.name} found {result['analysis']['result_count']} results with {result['confidence']:.2f} confidence"
            self.conversation_log.append(log_entry)
            logger.info(log_entry)
        
        self.iteration_count += 1
        return exploration_results
    
    def _collaborative_refinement(self, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Agents collaborate to refine findings."""
        logger.info("Phase 2: Collaborative refinement")
        
        # Identify agents that need help (low confidence or few results)
        agents_needing_help = []
        for agent_name, result in initial_results.items():
            if result['confidence'] < 0.6 or result['analysis']['needs_collaboration']:
                agents_needing_help.append(agent_name)
        
        # Facilitate collaboration
        for agent_name in agents_needing_help:
            agent = next(a for a in self.agents if a.name == agent_name)
            
            # Find the best helper agent
            helper_agent = self._find_best_helper(agent, initial_results)
            if helper_agent:
                # Create collaboration message
                help_message = AgentMessage(
                    sender=agent.name,
                    recipient=helper_agent.name,
                    content=self.goal,
                    message_type='request_help',
                    metadata={'original_confidence': initial_results[agent_name]['confidence']}
                )
                
                # Execute collaboration
                response = helper_agent.collaborate_with(agent, help_message)
                if response:
                    agent.collaboration_messages.append(response)
                    
                    # Agent tries again with shared context
                    refined_result = agent.process_query(self.goal, response.metadata)
                    initial_results[agent_name] = refined_result
                    
                    log_entry = f"{helper_agent.name} helped {agent.name} - new confidence: {refined_result['confidence']:.2f}"
                    self.conversation_log.append(log_entry)
                    logger.info(log_entry)
        
        self.iteration_count += 1
        return initial_results
    
    def _find_best_helper(self, agent: ResearchAgent, results: Dict[str, Any]) -> Optional[ResearchAgent]:
        """Find the best agent to help with a query."""
        best_helper = None
        best_confidence = 0
        
        for other_agent in self.agents:
            if other_agent.name != agent.name:
                other_confidence = results[other_agent.name]['confidence']
                if other_confidence > best_confidence:
                    best_confidence = other_confidence
                    best_helper = other_agent
        
        return best_helper if best_confidence > 0.6 else None
    
    def _final_synthesis(self, refined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Synthesize all findings."""
        logger.info("Phase 3: Final synthesis")
        
        # Collect all unique insights
        all_insights = []
        all_topics = set()
        total_results = 0
        weighted_confidence = 0
        
        for agent_name, result in refined_results.items():
            all_insights.extend(result['analysis']['insights'])
            all_topics.update(result['discovered_topics'])
            total_results += result['analysis']['result_count']
            weighted_confidence += result['confidence'] * result['analysis']['result_count']
        
        avg_confidence = weighted_confidence / max(total_results, 1)
        
        # Use the synthesizer agent for final synthesis
        synthesizer = next((a for a in self.agents if a.specialty == 'synthesizer'), self.agents[0])
        
        synthesis_query = f"Synthesize comprehensive answer for: {self.goal}. Key topics: {', '.join(list(all_topics)[:10])}"
        synthesis_result = synthesizer.process_query(synthesis_query)
        
        return {
            'synthesis_result': synthesis_result,
            'all_insights': list(set(all_insights)),  # Remove duplicates
            'all_topics': list(all_topics),
            'total_results': total_results,
            'avg_confidence': avg_confidence,
            'participating_agents': len(refined_results)
        }
    
    def _generate_final_report(self, synthesis: Dict[str, Any]) -> str:
        """Generate the final research report."""
        logger.info("Phase 4: Generating final report")
        
        # Create comprehensive context for Ollama
        context_parts = []
        
        # Gather all search results from all agents
        for agent in self.agents:
            if agent.history:
                latest_result = agent.history[-1]
                for search_result in latest_result.get('search_results', []):
                    context_parts.append(
                        f"From {search_result.metadata['filename']} (relevance: {search_result.relevance_score:.2f}):\n"
                        f"{search_result.content}\n"
                    )
        
        # Limit context to avoid token limits
        context = "\n".join(context_parts[:20])  # Top 20 most relevant pieces
        
        # Create enhanced prompt
        insights_text = "\n".join([f"- {insight}" for insight in synthesis['all_insights'][:10]])
        topics_text = ", ".join(synthesis['all_topics'][:15])
        
        prompt = f"""Based on comprehensive multi-agent research, provide a detailed answer to the following question.

Question: {self.goal}

Research Context:
{context}

Key Insights from Multiple AI Agents:
{insights_text}

Discovered Related Topics: {topics_text}

Research Statistics:
- Total results analyzed: {synthesis['total_results']}
- Average confidence: {synthesis['avg_confidence']:.2f}
- Agents participated: {synthesis['participating_agents']}

Instructions:
- Provide a comprehensive, well-structured answer
- Integrate insights from multiple perspectives
- Cite specific sources when possible
- If information is incomplete, state what additional research might be needed

Answer: """
        
        # Generate final answer using Ollama
        final_answer = self.search_system.call_ollama(prompt, temperature=0.3)
        
        # Add metadata to the report
        report = f"{final_answer}\n\n"
        report += f"Research Summary:\n"
        report += f"- Analyzed {synthesis['total_results']} document chunks\n"
        report += f"- Average confidence score: {synthesis['avg_confidence']:.2f}\n"
        report += f"- Discovered {len(synthesis['all_topics'])} related topics\n"
        report += f"- {synthesis['participating_agents']} AI agents collaborated\n"
        report += f"- Research completed in {self.iteration_count} iterations\n"
        
        # Add agent contributions
        report += f"\nAgent Contributions:\n"
        for agent in self.agents:
            if agent.history:
                searches = len(agent.history)
                topics = len(agent.discovered_topics)
                report += f"- {agent.name} ({agent.role}): {searches} searches, {topics} topics discovered\n"
        
        return report

class MarkdownSearchSystem:
    def __init__(self, 
                 output_folder: str = "output",
                 db_path: str = "./chroma_db",
                 collection_name: str = "markdown_docs",
                 ollama_base_url: str = "http://localhost:11434",
                 model_name: str = "mistral-small3.2",
                 embedding_model: str = "nomic-embed-text",
                 extensions: List[str] = None):
        
        self.output_folder = Path(output_folder)
        self.db_path = db_path
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.extensions = extensions or [".md", ".txt"]
        
        # Initialize ChromaDB with embedding function
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create embedding function for consistent embeddings
        try:
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url=f"{ollama_base_url}/api/embeddings",
                model_name=embedding_model
            )
        except Exception as e:
            print(f"Warning: Could not initialize Ollama embedding function: {e}")
            print("Using default ChromaDB embeddings")
            self.embedding_function = None
        
        # Handle existing collection with different embedding function
        try:
            # Try to get existing collection first
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if collection_name in existing_collections:
                print(f"Found existing collection: {collection_name}")
                # Get the existing collection without specifying embedding function
                self.collection = self.client.get_collection(name=collection_name)
                
                # Check if it's empty or has the right embedding function
                try:
                    count = self.collection.count()
                    print(f"Existing collection has {count} documents")
                    
                    if count > 0:
                        print("Using existing collection with its original embedding function")
                        # Don't override the embedding function for existing collections
                        self.embedding_function = None
                    else:
                        print("Collection is empty, will recreate with new embedding function")
                        self.client.delete_collection(name=collection_name)
                        self.collection = self.client.create_collection(
                            name=collection_name,
                            embedding_function=self.embedding_function,
                            metadata={"description": "Markdown documents collection"}
                        )
                except Exception as e:
                    print(f"Error accessing existing collection: {e}")
                    print("Using existing collection as-is")
            else:
                # Create new collection with embedding function
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"description": "Markdown documents collection"}
                )
        except Exception as e:
            print(f"Error with collection setup: {e}")
            print("Falling back to get_or_create without embedding function")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Markdown documents collection"}
            )
            self.embedding_function = None
        
        print(f"Initialized ChromaDB at: {db_path}")
        print(f"Collection: {collection_name}")
        print(f"Embedding model: {embedding_model}")
    
    def extract_content(self, file_path: Path) -> Dict[str, str]:
        """Extract content and metadata from a markdown or text file."""
        try:
            # Handle frontmatter for markdown files
            if file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                    content = post.content
                    fm_metadata = post.metadata
                
                # Extract title from frontmatter or first heading
                title = fm_metadata.get('title')
                if not title:
                    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                    title = title_match.group(1) if title_match else file_path.stem
                
                # Get date from frontmatter or filename
                date_str = fm_metadata.get('date')
                if date_str and hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                elif not date_str:
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path.name)
                    date_str = date_match.group(1) if date_match else "unknown"
                
                # Merge frontmatter with file metadata
                extra_metadata = {k: str(v) for k, v in fm_metadata.items() 
                                if k not in ['title', 'date']}
            else:
                # Handle plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                title = file_path.stem
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path.name)
                date_str = date_match.group(1) if date_match else "unknown"
                extra_metadata = {}
            
            # Create metadata
            metadata = {
                "title": title,
                "filename": file_path.name,
                "filepath": str(file_path),
                "date": date_str,
                "year": file_path.parent.parent.name if len(file_path.parts) > 2 else "unknown",
                "month": file_path.parent.name if len(file_path.parts) > 1 else "unknown",
                "file_size": len(content),
                "created_at": datetime.now().isoformat(),
                "file_extension": file_path.suffix
            }
            
            # Add any extra metadata from frontmatter
            metadata.update(extra_metadata)
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def chunk_content_smart(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split content into overlapping chunks using sentence boundaries."""
        if len(content) <= chunk_size:
            return [content]
        
        try:
            # Use NLTK for better sentence splitting
            sentences = sent_tokenize(content)
        except Exception:
            # Fallback to simple splitting if NLTK fails
            sentences = re.split(r'[.!?]+\s+', content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap from previous chunk
                    words = current_chunk.split()
                    overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    # Single sentence is too long, split it
                    if len(sentence) > chunk_size:
                        words = sentence.split()
                        for i in range(0, len(words), chunk_size//10):
                            chunk_words = words[i:i + chunk_size//10]
                            chunks.append(" ".join(chunk_words))
                    else:
                        current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_chunk_id(self, filepath: str, chunk_content: str, chunk_index: int = 0) -> str:
        """Generate a unique ID for a document chunk based on content hash."""
        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
        path_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
        return f"{path_hash}_{chunk_index}_{content_hash}"
    
    def process_files_batch(self, batch_size: int = 50):
        """Process all supported files in the folder structure with batch insertion."""
        if not self.output_folder.exists():
            print(f"Output folder {self.output_folder} does not exist!")
            return
        
        processed_count = 0
        skipped_count = 0
        
        # Find all supported files
        all_files = []
        for ext in self.extensions:
            all_files.extend(list(self.output_folder.glob(f"**/*{ext}")))
        
        print(f"Found {len(all_files)} files to process...")
        
        # Process files in batches
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        for file_path in all_files:
            try:
                # Extract content and metadata
                doc_data = self.extract_content(file_path)
                if not doc_data:
                    skipped_count += 1
                    continue
                
                content = doc_data["content"]
                metadata = doc_data["metadata"]
                
                # Check if document already exists (by filepath)
                existing = self.collection.get(
                    where={"filepath": str(file_path)}
                )
                
                if existing['ids']:
                    print(f"Skipping {file_path.name} (already processed)")
                    skipped_count += 1
                    continue
                
                # Chunk the content
                chunks = self.chunk_content_smart(content)
                
                # Prepare batch data for each chunk
                for i, chunk in enumerate(chunks):
                    doc_id = self.generate_chunk_id(str(file_path), chunk, i)
                    
                    # Add chunk-specific metadata
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk),
                        "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8],
                        "embedding_added_at": datetime.now().isoformat()
                    })
                    
                    batch_documents.append(chunk)
                    batch_metadatas.append(chunk_metadata)
                    batch_ids.append(doc_id)
                
                processed_count += 1
                print(f"Prepared: {file_path.name} ({len(chunks)} chunks)")
                
                # Insert batch when it reaches batch_size
                if len(batch_documents) >= batch_size:
                    self.collection.add(
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    print(f"Inserted batch of {len(batch_documents)} documents")
                    batch_documents.clear()
                    batch_metadatas.clear()
                    batch_ids.clear()
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped_count += 1
        
        # Insert remaining documents
        if batch_documents:
            self.collection.add(
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"Inserted final batch of {len(batch_documents)} documents")
        
        print(f"\nProcessing complete!")
        print(f"Processed: {processed_count} files")
        print(f"Skipped: {skipped_count} files")
        print(f"Total documents in collection: {self.collection.count()}")
    
    def search_documents(self, query: str, n_results: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        """Search for relevant documents using semantic similarity with optional filtering."""
        try:
            query_params = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            results = self.collection.query(**query_params)
            
            search_results = []
            for i in range(len(results['ids'][0])):
                search_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]  # Convert distance to relevance
                })
            
            return search_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API with the given prompt."""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error calling Ollama: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"
    
    def answer_question(self, question: str, n_results: int = 5, 
                       date_filter: str = None, min_relevance: float = 0.3) -> str:
        """Answer a question using retrieved context and Ollama with optional date filtering."""
        print(f"Searching for: '{question}'")
        
        # Prepare metadata filter
        filter_metadata = {}
        if date_filter:
            # Simple date filtering - you could expand this
            filter_metadata["date"] = date_filter
        
        # Search for relevant documents
        search_results = self.search_documents(question, n_results, filter_metadata)
        
        if not search_results:
            return "No relevant documents found for your question."
        
        # Filter by relevance score
        relevant_results = [r for r in search_results if r['relevance_score'] >= min_relevance]
        
        if not relevant_results:
            return f"No documents found with relevance score >= {min_relevance}. Try a different question or lower the threshold."
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(relevant_results, 1):
            metadata = result['metadata']
            relevance = result['relevance_score']
            context_parts.append(
                f"Document {i} (from {metadata['filename']}, {metadata['date']}, relevance: {relevance:.2f}):\n"
                f"{result['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create enhanced prompt for Ollama
        prompt = f"""Based on the following documents, please answer the question accurately and comprehensively. 
Use only the information provided in the documents. If the documents don't contain enough information to fully answer the question, please state that clearly.

Question: {question}

Relevant Documents:
{context}

Instructions:
- Provide a clear, well-structured answer

Answer: """
        
        print("Generating answer with Ollama...")
        answer = self.call_ollama(prompt)
        
        # Add enhanced source information
        sources = []
        for result in relevant_results:
            metadata = result['metadata']
            relevance = result['relevance_score']
            sources.append(f"- {metadata['filename']} ({metadata['date']}) - relevance: {relevance:.2f}")
        
        return f"{answer}\n\nSources ({len(sources)} documents):\n" + "\n".join(sources)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection."""
        try:
            count = self.collection.count()
            if count == 0:
                return {"total_documents": 0}
            
            # Get sample of metadata to analyze
            sample = self.collection.get(limit=min(100, count), include=["metadatas"])
            
            dates = []
            file_types = {}
            years = {}
            
            for metadata in sample['metadatas']:
                # Count file types
                ext = metadata.get('file_extension', 'unknown')
                file_types[ext] = file_types.get(ext, 0) + 1
                
                # Count years
                year = metadata.get('year', 'unknown')
                years[year] = years.get(year, 0) + 1
                
                # Collect dates
                date = metadata.get('date')
                if date and date != 'unknown':
                    dates.append(date)
            
            return {
                "total_documents": count,
                "file_types": file_types,
                "years": years,
                "date_range": {
                    "earliest": min(dates) if dates else "unknown",
                    "latest": max(dates) if dates else "unknown"
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def interactive_search(self):
        """Start an interactive search session with enhanced features."""
        print("\n=== Enhanced Multi-Agent Research System ===")
        print("Commands:")
        print("  - Type your research question to start multi-agent search")
        print("  - 'single <question>' for single-agent search")
        print("  - 'stats' to see collection statistics")
        print("  - 'agents' to see available agent types")
        print("  - 'help' to see this help")
        print("  - 'quit' to exit")
        print("-" * 50)
        
        # Show initial stats
        stats = self.get_collection_stats()
        if 'error' not in stats:
            print(f"Collection loaded: {stats['total_documents']} documents")
            if stats['total_documents'] > 0:
                print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
                print(f"File types: {', '.join(f'{k}({v})' for k, v in stats['file_types'].items())}")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - <question> - Multi-agent collaborative research")
                    print("  - single <question> - Traditional single search")
                    print("  - agents - List available agent specialties")
                    print("  - stats - Show collection statistics") 
                    print("  - quit - Exit the program")
                    continue
                
                if user_input.lower() == 'agents':
                    print("\nAvailable Agent Specialties:")
                    print("  - Contextualizer: Finds background and definitions")
                    print("  - Synthesizer: Summarizes and combines information")
                    print("  - Validator: Verifies and validates claims")
                    print("  - Explorer: Discovers related topics and connections")
                    print("  - Temporal: Analyzes chronological information")
                    print("  - Technical: Focuses on technical details")
                    continue
                
                if user_input.lower() == 'stats':
                    stats = self.get_collection_stats()
                    print(f"\nCollection Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                
                if user_input.startswith('single '):
                    question = user_input[7:].strip()
                    if question:
                        print("\nUsing traditional single-agent search...")
                        answer = self.answer_question(question)
                        print(f"\nAnswer:\n{answer}")
                    continue
                
                if not user_input:
                    continue
                
                # Multi-agent research
                print(f"\n🤖 Initiating multi-agent research for: '{user_input}'")
                print("=" * 60)
                
                orchestrator = ResearchOrchestrator(self, user_input)
                result = orchestrator.run()
                
                if result['status'] == 'completed':
                    print(f"\n📊 Research Results:")
                    print(f"Final Report:\n{result['final_report']}")
                    print("\n" + "=" * 60)
                    print(f"Research completed in {result['iterations']} iterations")
                    print(f"Agent contributions: {result['agent_contributions']}")
                    print(f"Topics discovered: {result['total_discovered_topics']}")
                else:
                    print(f"❌ Research failed: {result.get('error', 'Unknown error')}")
                    if result.get('partial_results'):
                        print(f"Partial results: {result['partial_results']}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main function to run the enhanced search system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent Markdown Search System with ChromaDB and Ollama")
    parser.add_argument("--output-folder", default="output", help="Path to output folder containing files")
    parser.add_argument("--db-path", default="./chroma_db", help="Path for ChromaDB storage")
    parser.add_argument("--collection", default="markdown_docs", help="ChromaDB collection name")
    parser.add_argument("--model", default="mistral", help="Ollama model name")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--extensions", nargs='+', default=[".md", ".txt"], help="File extensions to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--process", action="store_true", help="Process files into ChromaDB")
    parser.add_argument("--search", action="store_true", help="Start interactive multi-agent search")
    parser.add_argument("--question", help="Ask a single question using multi-agent research")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--reset-collection", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations for multi-agent research")
    
    args = parser.parse_args()
    
    # Handle collection reset
    if args.reset_collection:
        print(f"Resetting collection '{args.collection}'...")
        try:
            client = chromadb.PersistentClient(path=args.db_path)
            try:
                client.delete_collection(name=args.collection)
                print(f"Deleted existing collection: {args.collection}")
            except Exception as e:
                print(f"Collection didn't exist or couldn't be deleted: {e}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            return
    
    # Initialize the search system
    search_system = MarkdownSearchSystem(
        output_folder=args.output_folder,
        db_path=args.db_path,
        collection_name=args.collection,
        model_name=args.model,
        embedding_model=args.embedding_model,
        extensions=args.extensions
    )
    
    if args.process:
        print("Processing files...")
        search_system.process_files_batch(batch_size=args.batch_size)
    
    elif args.stats:
        stats = search_system.get_collection_stats()
        print("\nCollection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.question:
        print(f"🤖 Multi-Agent Research: '{args.question}'")
        print("=" * 60)
        
        orchestrator = ResearchOrchestrator(search_system, args.question, max_iterations=args.max_iterations)
        result = orchestrator.run()
        
        if result['status'] == 'completed':
            print(f"\n📊 Research Results:")
            print(result['final_report'])
            print("\n" + "=" * 60)
            print(f"Research completed in {result['iterations']} iterations")
            print(f"Agent contributions: {result['agent_contributions']}")
            print(f"Topics discovered: {result['total_discovered_topics']}")
        else:
            print(f"❌ Research failed: {result.get('error', 'Unknown error')}")
    
    elif args.search:
        search_system.interactive_search()
    
    else:
        print("Enhanced Multi-Agent Markdown Search System")
        print("\nUsage examples:")
        print("  python analyze.py --process                    # Process files into ChromaDB")
        print("  python analyze.py --search                     # Interactive multi-agent search")
        print("  python analyze.py --stats                      # Show collection stats")
        print("  python analyze.py --question 'What happened in January 2024?'  # Multi-agent research")
        print("  python analyze.py --process --search           # Process then search")
        print("  python analyze.py --extensions .md .txt .rst   # Process multiple file types")
        print("  python analyze.py --reset-collection           # Delete existing collection")
        print("  python analyze.py --reset-collection --process # Reset and reprocess")
        print("  python analyze.py --max-iterations 10 --question 'Complex research question'")


if __name__ == "__main__":
    main()