# AWS Bedrock QA with LangChain and FAISS

This project demonstrates a **question-answering system** using **AWS Bedrock**, **LangChain**, and **FAISS** for document retrieval. Users can query a set of documents, and the system retrieves relevant chunks and generates answers using a Bedrock LLM.

## Features

- Integrates **AWS Bedrock LLMs** (e.g., Claude, Titan, or AI21) with LangChain.
- Uses **FAISS** for fast vector-based document retrieval.
- Supports **large documents** with `map_reduce` or `refine` chains to avoid token limits.
- Easily extendable for different models or document sources.

## Project Structure


AWSBEDROCK/
│
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── data/               # Directory for raw documents
├── faiss_index/        # FAISS vector store
└── README.md           # Project documentation


## Requirements

- Python 3.10+
- AWS account with **Bedrock access**
- Virtual environment recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your AWS credentials (via environment variables or `~/.aws/credentials`).


## Usage

1. Prepare your documents and store them in the `data/` folder.
2. Build FAISS index (if not already built) and store it in `faiss_index/`.
3. Run the app:

```cmd
python app.py
```

4. Enter your query in the Streamlit app (or console if configured) to get answers.

## Configuration

- **Model selection**: Update the `model_id` in `app.py` to your preferred Bedrock model:

```python
llm = Bedrock(
    model_id="anthropic.claude-v2",
    region_name="us-east-1",
    temperature=0.7,
    max_tokens=500
)
```

- **Retriever settings**: Limit the number of documents retrieved to avoid token limits:

```python
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
```

- **Chain type**: Use `"map_reduce"` or `"refine"` instead of `"stuff"` to handle large contexts safely.

## Notes

- AWS Bedrock models have **maximum token limits** (e.g., Claude v2: 8192 tokens). Large documents may need chunking.
- `.gitignore` excludes `data/` and `faiss_index/` by default to avoid pushing large files or sensitive data.
- Customize prompts in `app.py` for different behavior.

## References

- [LangChain Documentation](https://www.langchain.com/docs/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [FAISS Library](https://github.com/facebookresearch/faiss)
