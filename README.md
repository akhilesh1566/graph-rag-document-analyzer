# 🤖 Intelligent Document Analyst

**A Graph RAG-powered Document Q&A System**

Transform any PDF document into an interactive knowledge graph for advanced question-answering that goes beyond simple keyword matching.

## 🌟 What Makes This Special?

Unlike traditional RAG systems that use vector similarity search over disconnected text chunks, this system:

- **Builds Knowledge Graphs**: Extracts entities and relationships from documents
- **Understands Context**: Answers complex questions requiring multi-hop reasoning
- **Provides Insights**: Reveals hidden connections and patterns in your documents

### Example Capabilities

**Traditional RAG:** "What is Product X?" → Simple fact retrieval

**Graph RAG:** "How did the supply chain issues mentioned in Q3 impact the launch timeline of Product X?" → Multi-hop reasoning across entities and relationships

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Desktop** (Community Edition)
3. **OpenAI API Key**

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd graph-rag-document-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Neo4j**
   - Download and install [Neo4j Desktop](https://neo4j.com/download/)
   - Create a new project and database
   - Start the database and note the credentials

5. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your credentials:
   # OPENAI_API_KEY=sk-your-api-key
   # NEO4J_URI=neo4j://localhost:7687
   # NEO4J_USERNAME=neo4j
   # NEO4J_PASSWORD=your-password
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

### Web Interface

1. **Upload PDF**: Use the web interface to upload your document
2. **Build Graph**: Click "Build Knowledge Graph" to process the document
3. **Ask Questions**: Use natural language to query your document

### Jupyter Notebook

For development and experimentation:
```bash
jupyter notebook notebooks/1_exploration.ipynb
```

### Example Questions

- **Simple**: "Who is mentioned in the document?"
- **Relational**: "What products did [Person] work on?"
- **Multi-hop**: "How did the Q4 results affect the strategy for Product X?"
- **Analytical**: "What were the key risks and their mitigation strategies?"

## 🏗️ Architecture

```
PDF Document → Document Loader → Knowledge Graph Builder → Neo4j Database
                                                                ↓
User Question → Query Translator → Graph Query → Context Retrieval → LLM Generator → Answer
```

### Key Components

- **`src/loader.py`**: PDF loading and text chunking
- **`src/graph_builder.py`**: Knowledge graph construction
- **`src/chains.py`**: RAG chains and query processing
- **`app.py`**: Streamlit web interface

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `NEO4J_URI` | Neo4j database URI | `neo4j://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `OPENAI_EXTRACTION_MODEL` | Model for query generation | `gpt-4o-mini` |
| `OPENAI_GENERATION_MODEL` | Model for answer generation | `gpt-4o` |

### Customization

- **Chunk size**: Modify in `DocumentLoader` (default: 1000 chars)
- **LLM models**: Configure via environment variables
- **Graph limits**: Set max nodes/relationships in config

## 🛠️ Development

### Project Structure

```
graph-rag-document-analyzer/
├── src/                    # Source code
│   ├── loader.py          # Document loading
│   ├── graph_builder.py   # Graph construction
│   └── chains.py          # RAG chains
├── data/                  # Document storage
├── notebooks/             # Jupyter notebooks
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
└── .env.template         # Environment template
```

### Running Tests

Development and testing via Jupyter notebook:
```bash
jupyter notebook notebooks/1_exploration.ipynb
```

### Adding Features

The modular architecture makes it easy to extend:
- Add new document loaders
- Customize entity extraction
- Implement new query types
- Enhance the UI

## 📊 Performance

- **Document Processing**: ~1-2 minutes per 100 pages
- **Query Response**: ~3-5 seconds per question
- **Graph Storage**: Efficient Neo4j indexing
- **Memory Usage**: Optimized chunking strategy

## 🔒 Security

- Environment variable management for API keys
- No hardcoded credentials
- Secure Neo4j authentication
- Input validation and error handling

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Error**
   - Ensure Neo4j is running
   - Check credentials in `.env`
   - Verify URI format

2. **OpenAI API Error**
   - Verify API key is correct
   - Check account credits/limits
   - Ensure proper model access

3. **Document Processing Error**
   - Check PDF file format
   - Ensure file is readable
   - Try smaller chunk sizes

### Debug Mode

Enable verbose logging by setting `verbose=True` in the configuration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG framework
- [Neo4j](https://neo4j.com/) for graph database
- [OpenAI](https://openai.com/) for language models
- [Streamlit](https://streamlit.io/) for web interface