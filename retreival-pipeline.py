from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

persistent_directory = "dbv2/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model
    )

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


stored_data = vectorstore.get(include=["documents", "metadatas"])

documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(stored_data["documents"], stored_data["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3


hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight to vector and keyword search
)




query = "What are the two main components of the Transformer architecture? "
chunks = hybrid_retriever.invoke(query)



def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# Usage
final_answer = generate_final_answer(chunks, query)
print(final_answer)








def recall_at_k(retriever, evaluation_set, k=3):

    hits = 0

    for item in evaluation_set:

        question = item["question"]
        expected_ids = set(item["expected_chunk_ids"])

        retrieved_docs = retriever.invoke(question)
        retrieved_docs = retrieved_docs[:k]

        retrieved_ids = {
            doc.metadata["chunk_id"]
            for doc in retrieved_docs
        }

        if 40 in set(item["expected_chunk_ids"]):
            print("/"*13)
            print(retrieved_ids)
            print("/"*13)

        if expected_ids.intersection(retrieved_ids):
            hits += 1

    return hits / len(evaluation_set)


def mean_reciprocal_rank(retriever, evaluation_set):

    total_rr = 0

    for item in evaluation_set:

        expected_ids = set(item["expected_chunk_ids"])

        docs = retriever.invoke(item["question"])

        rr = 0

        for rank, doc in enumerate(docs, start=1):

            if doc.metadata["chunk_id"] in expected_ids:
                rr = 1 / rank
                break

        total_rr += rr

    return total_rr / len(evaluation_set)




evaluation_set2 = [
    {
        "question": "What architecture did the Transformer introduce to replace recurrent and convolutional networks in sequence transduction tasks?",
        "expected_chunk_ids": [1]
    },
    {
        "question": "Why do recurrent neural networks limit parallelization during training?",
        "expected_chunk_ids": [2]
    },
    {
        "question": "How does the Transformer reduce the number of operations required to relate distant positions compared with ConvS2S and ByteNet?",
        "expected_chunk_ids": [3]
    },
    {
        "question": "What are the main components shown in the Transformer encoder-decoder architecture?",
        "expected_chunk_ids": [4]
    },
    {
        "question": "How many layers are used in the Transformer encoder stack, and what are the two sub-layers in each encoder layer?",
        "expected_chunk_ids": [5]
    },
    {
        "question": "What additional sub-layer is included in each decoder layer that is not present in encoder layers?",
        "expected_chunk_ids": [5]
    },
    {
        "question": "What is the purpose of an attention function in the Transformer architecture?",
        "expected_chunk_ids": [6]
    },
    {
        "question": "How is Scaled Dot-Product Attention computed from queries, keys, and values?",
        "expected_chunk_ids": [7]
    },
    {
        "question": "Why is the dot product scaled by the square root of dk in Scaled Dot-Product Attention?",
        "expected_chunk_ids": [7]
    },
    {
        "question": "What is the main idea behind Multi-Head Attention?",
        "expected_chunk_ids": [8]
    },
    {
        "question": "How does Multi-Head Attention help the model learn information from different representation subspaces?",
        "expected_chunk_ids": [8]
    },
    {
        "question": "What are the three different applications of multi-head attention used in the Transformer model?",
        "expected_chunk_ids": [9]
    },
    {
        "question": "Why must decoder self-attention prevent positions from attending to future tokens?",
        "expected_chunk_ids": [9]
    },
    {
        "question": "What is the structure of the position-wise feed-forward network used in Transformer layers?",
        "expected_chunk_ids": [10]
    },
    {
        "question": "What are the values of dmodel and dff used in the Transformer feed-forward networks?",
        "expected_chunk_ids": [10]
    },
    {
        "question": "How are embeddings and the softmax layer implemented in the Transformer model?",
        "expected_chunk_ids": [11]
    },
    {
        "question": "Which layer type has O(1) sequential operations and O(1) maximum path length according to the complexity comparison table?",
        "expected_chunk_ids": [12]
    },
    {
        "question": "Why are positional encodings required in the Transformer architecture?",
        "expected_chunk_ids": [13]
    },
    {
        "question": "What mathematical functions are used to construct the Transformer's positional encodings?",
        "expected_chunk_ids": [13]
    },
    {
        "question": "What three criteria are used to compare self-attention with recurrent and convolutional layers?",
        "expected_chunk_ids": [14]
    },
    {
        "question": "What evidence suggests that self-attention may improve model interpretability?",
        "expected_chunk_ids": [15]
    },
    {
        "question": "What datasets and vocabulary sizes were used for the English-German and English-French translation experiments?",
        "expected_chunk_ids": [15]
    },
    {
        "question": "What optimizer and learning-rate schedule were used to train the Transformer models?",
        "expected_chunk_ids": [16]
    },
    {
        "question": "What dropout rate and label smoothing value were used for the base Transformer model?",
        "expected_chunk_ids": [19]
    },
    {
        "question": "What BLEU scores did the Transformer achieve on the WMT 2014 English-to-German and English-to-French translation tasks?",
        "expected_chunk_ids": [20]
    }
]

evaluation_set = [
    {
        "question": "What is the title of the paper that introduced the Transformer architecture?",
        "expected_chunk_ids": [0]
    },
    {
        "question": "What key innovation allows the Transformer to eliminate both recurrence and convolution?",
        "expected_chunk_ids": [1]
    },
    {
        "question": "Why do recurrent models become difficult to parallelize during training?",
        "expected_chunk_ids": [2]
    },
    {
        "question": "How does the Transformer improve the ability to model long-range dependencies compared to ConvS2S and ByteNet?",
        "expected_chunk_ids": [3]
    },
    {
        "question": "What are the primary building blocks used in the Transformer encoder and decoder architecture?",
        "expected_chunk_ids": [4]
    },
    {
        "question": "How many layers are stacked in the Transformer encoder, and what sub-layers does each encoder layer contain?",
        "expected_chunk_ids": [5]
    },
    {
        "question": "What does an attention function map from and to in the Transformer model?",
        "expected_chunk_ids": [6]
    },
    {
        "question": "What is the formula used for Scaled Dot-Product Attention?",
        "expected_chunk_ids": [7]
    },
    {
        "question": "Why does the Transformer project queries, keys, and values multiple times in Multi-Head Attention?",
        "expected_chunk_ids": [8]
    },
    {
        "question": "What are the three distinct ways multi-head attention is applied within the Transformer architecture?",
        "expected_chunk_ids": [9]
    },
    {
        "question": "What activation function is used inside the position-wise feed-forward network?",
        "expected_chunk_ids": [10]
    },
    {
        "question": "How are embedding weights reused between the embedding layers and output projection layer?",
        "expected_chunk_ids": [11]
    },
    {
        "question": "Which layer type has a maximum path length of O(1) according to the complexity comparison table?",
        "expected_chunk_ids": [12]
    },
    {
        "question": "Why are positional encodings necessary in the Transformer architecture?",
        "expected_chunk_ids": [13]
    },
    {
        "question": "What three criteria are used to compare self-attention against recurrent and convolutional layers?",
        "expected_chunk_ids": [14]
    },
    {
        "question": "What evidence suggests that different attention heads learn different linguistic functions?",
        "expected_chunk_ids": [15]
    },
    {
        "question": "How many NVIDIA P100 GPUs were used to train the Transformer models described in the paper?",
        "expected_chunk_ids": [16]
    },
    {
        "question": "What is the purpose of label smoothing during Transformer training?",
        "expected_chunk_ids": [19]
    },
    {
        "question": "What BLEU score did the Transformer (big) achieve on the WMT 2014 English-to-German translation task?",
        "expected_chunk_ids": [20]
    },
    {
        "question": "Why did the authors perform model variation experiments on the Transformer architecture?",
        "expected_chunk_ids": [21]
    },
    {
        "question": "What BLEU score and parameter count are reported for the base Transformer model in the architecture variation results?",
        "expected_chunk_ids": [22]
    },
    {
        "question": "What effect did reducing the attention key dimension dk have on model quality?",
        "expected_chunk_ids": [23]
    },
    {
        "question": "How many training sentences from the Penn Treebank WSJ dataset were used for the English constituency parsing experiments?",
        "expected_chunk_ids": [24]
    },
    {
        "question": "What parsing accuracy did the 4-layer Transformer achieve in the semi-supervised WSJ setting?",
        "expected_chunk_ids": [25]
    },
    {
        "question": "What future research directions for attention-based models are proposed in the paper's conclusion?",
        "expected_chunk_ids": [27]
    }
]




retrievers = {
    "BM25": bm25_retriever,
    "Vector": vector_retriever,
    "Hybrid": hybrid_retriever
}

for name, retriever in retrievers.items():

    recall = recall_at_k(retriever, evaluation_set)
    mrr = mean_reciprocal_rank(retriever, evaluation_set)

    print(
        f"{name}: "
        f"Recall={recall:.3f} "
        f"MRR={mrr:.3f}"
    )






