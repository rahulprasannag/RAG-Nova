'''

langchain_cohere

'''

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import time
from cohere.errors import TooManyRequestsError

load_dotenv()

persistent_directory = "dbv2/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embedding_model
    )

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


stored_data = vectorstore.get(include=["documents", "metadatas"])

documents = [
    Document(page_content=text, metadata=meta)
    for text, meta in zip(stored_data["documents"], stored_data["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10


hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight to vector and keyword search
)



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
# final_answer = generate_final_answer(chunks, query)
# print(final_answer)

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks







def rerank_with_retry(docs, query, max_retries=5):

    for attempt in range(max_retries):
        try:
            return reranker.compress_documents(docs, query)

        except TooManyRequestsError:
            wait_time = 60
            print(f"Rate limited. Sleeping {wait_time}s...")
            time.sleep(wait_time)

    return docs












# llm = ChatOpenAI(model="gpt-4o", temperature=0)


# # Pydantic model for structured output
# class QueryVariations(BaseModel):
#     queries: List[str]


# # Original query
# original_query = "How does Tesla make money?"
# print(f"Original Query: {original_query}\n")

# # ──────────────────────────────────────────────────────────────────
# # Step 1: Generate Multiple Query Variations
# # ──────────────────────────────────────────────────────────────────

# llm_with_tools = llm.with_structured_output(QueryVariations)

# prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

# Original query: {original_query}

# Return 3 alternative queries that rephrase or approach the same question from different angles."""

# response = llm_with_tools.invoke(prompt)
# query_variations = response.queries

# print("Generated Query Variations:")
# for i, variation in enumerate(query_variations, 1):
#     print(f"{i}. {variation}")

# print("\n" + "="*60)






# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

# all_retrieval_results = []  # Store all results for RRF

# for i, query in enumerate(query_variations, 1):
#     print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
#     docs = hybrid_retriever.invoke(query)
#     all_retrieval_results.append(docs)  # Store for RRF calculation
    
#     print(f"Retrieved {len(docs)} documents:\n")
    
#     for j, doc in enumerate(docs, 1):
#         print(f"Document {j}:")
#         print(f"{doc.page_content[:150]}...\n")
    
#     print("-" * 50)

# print("\n" + "="*60)
# print("Multi-Query Retrieval Complete!")
# print("Notice how different query variations retrieved different documents.")



# # ──────────────────────────────────────────────────────────────────
# # Step 3: Apply Reciprocal Rank Fusion (RRF)
# # ──────────────────────────────────────────────────────────────────

# # Apply RRF to our retrieval results
# fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)









# query = "What are the two main components of the Transformer architecture? "
# chunks = hybrid_retriever.invoke(query) 
# print("8"*13)
# print(len(chunks))
# print("8"*13)

reranker = CohereRerank(model="rerank-english-v3.0", top_n=7)
# chunks = reranker.compress_documents(chunks,query )

















def recall_at_k(retriever, evaluation_set, k=3):

    hits = 0

    for item in evaluation_set:

        question = item["question"]
        expected_ids = set(item["expected_chunk_ids"])

        retrieved_docs = retriever.invoke(question)
        retrieved_docs = rerank_with_retry(retrieved_docs,item["question"] )
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
        docs = rerank_with_retry(docs,item["question"] )

        rr = 0

        for rank, doc in enumerate(docs, start=1):

            if doc.metadata["chunk_id"] in expected_ids:
                rr = 1 / rank
                break

        total_rr += rr

    return total_rr / len(evaluation_set)


evaluation_set_for_multiquery = [
    {
        "question1": "What novel neural network architecture was introduced by the Transformer paper as an alternative to recurrent and convolution-based sequence transduction models?",
        "question2": "Which architecture did the Transformer propose to eliminate the need for RNNs and CNNs in machine translation and other sequence-to-sequence tasks?",
        "question3": "In the 'Attention Is All You Need' paper, what model architecture replaced traditional recurrent and convolutional networks for sequence transduction?",
        "expected_chunk_ids": [1]
    },

    {
        "question1": "What aspect of recurrent neural networks makes parallel training difficult?",
        "question2": "Why can't RNN computations be fully parallelized during model training?",
        "question3": "How does the sequential nature of recurrent networks restrict parallel processing?",
        "expected_chunk_ids": [2]
    },

    {
        "question1": "How does the Transformer improve long-range dependency modeling efficiency compared to ConvS2S and ByteNet?",
        "question2": "Why does self-attention require fewer operations than ConvS2S and ByteNet when connecting distant positions?",
        "question3": "What advantage does the Transformer have over ConvS2S and ByteNet in relating tokens that are far apart?",
        "expected_chunk_ids": [3]
    },

    {
        "question1": "What modules make up the encoder-decoder structure of the Transformer architecture?",
        "question2": "Which components are included in the Transformer model diagram?",
        "question3": "What are the key building blocks of the Transformer's encoder and decoder?",
        "expected_chunk_ids": [4]
    },

    {
        "question1": "How many encoder layers are stacked in the Transformer, and what sub-layers does each contain?",
        "question2": "What is the composition of a Transformer encoder layer and how many times is it repeated?",
        "question3": "Describe the encoder stack depth and the two sub-components present in every encoder block.",
        "expected_chunk_ids": [5]
    },

    {
        "question1": "What extra component exists in decoder layers that encoder layers do not have?",
        "question2": "Which additional sub-layer distinguishes the Transformer decoder from the encoder?",
        "question3": "What unique attention mechanism is included in each decoder layer beyond the encoder structure?",
        "expected_chunk_ids": [5]
    },

    {
        "question1": "What role does the attention mechanism play within the Transformer model?",
        "question2": "Why is an attention function used in Transformer architectures?",
        "question3": "How does attention help process information in the Transformer?",
        "expected_chunk_ids": [6]
    },

    {
        "question1": "How are queries, keys, and values combined to compute Scaled Dot-Product Attention?",
        "question2": "What steps are involved in calculating Scaled Dot-Product Attention from Q, K, and V matrices?",
        "question3": "Explain the computation process for Scaled Dot-Product Attention using queries, keys, and values.",
        "expected_chunk_ids": [7]
    },

    {
        "question1": "Why is attention scaled by √dk in the Scaled Dot-Product Attention formula?",
        "question2": "What problem does dividing by the square root of dk solve in attention calculations?",
        "question3": "Why does the Transformer normalize dot products using the square root of the key dimension?",
        "expected_chunk_ids": [7]
    },

    {
        "question1": "What is the core concept behind the Multi-Head Attention mechanism?",
        "question2": "How does Multi-Head Attention differ from using a single attention operation?",
        "question3": "What idea motivates the use of multiple attention heads in the Transformer?",
        "expected_chunk_ids": [8]
    },

    {
        "question1": "How do multiple attention heads capture information from different representation spaces?",
        "question2": "Why does Multi-Head Attention enable learning from diverse subspaces?",
        "question3": "How does the Transformer use multiple heads to attend to different feature representations?",
        "expected_chunk_ids": [8]
    },

    {
        "question1": "In which three places is Multi-Head Attention applied within the Transformer architecture?",
        "question2": "What are the three uses of multi-head attention in the encoder-decoder model?",
        "question3": "Where does the Transformer employ multi-head attention throughout the network?",
        "expected_chunk_ids": [9]
    },

    {
        "question1": "Why is future-token masking required in decoder self-attention?",
        "question2": "What is the reason decoder attention cannot access subsequent positions?",
        "question3": "Why must the Transformer decoder block attention to future words during training?",
        "expected_chunk_ids": [9]
    },

    {
        "question1": "What architecture is used for the Transformer's position-wise feed-forward network?",
        "question2": "How is the feed-forward layer inside each Transformer block structured?",
        "question3": "Describe the design of the position-wise fully connected network in Transformer layers.",
        "expected_chunk_ids": [10]
    },

    {
        "question1": "What values are assigned to d_model and d_ff in the Transformer feed-forward layers?",
        "question2": "Which dimensions are used for dmodel and dff in the original Transformer architecture?",
        "question3": "What are the specified model and feed-forward dimensions in the Transformer's FFN?",
        "expected_chunk_ids": [10]
    },

    {
        "question1": "How are token embeddings and output softmax weights handled in the Transformer?",
        "question2": "What implementation strategy is used for embeddings and the softmax layer in the Transformer model?",
        "question3": "How does the Transformer share or utilize embeddings with the softmax output layer?",
        "expected_chunk_ids": [11]
    },

    {
        "question1": "Which neural layer achieves constant sequential operations and constant maximum path length in the complexity analysis?",
        "question2": "According to the complexity table, what layer type has O(1) sequential computation and O(1) path length?",
        "question3": "What architecture exhibits both O(1) sequential operations and O(1) maximum dependency path length?",
        "expected_chunk_ids": [12]
    },

    {
        "question1": "Why does the Transformer require positional encoding information?",
        "question2": "What problem do positional encodings solve in self-attention models?",
        "question3": "Why must token position information be added to Transformer inputs?",
        "expected_chunk_ids": [13]
    },

    {
        "question1": "Which mathematical functions are used to generate positional encodings in the Transformer?",
        "question2": "How are sinusoidal positional encodings constructed in the original Transformer model?",
        "question3": "What formulas underlie the Transformer's positional representation scheme?",
        "expected_chunk_ids": [13]
    },

    {
        "question1": "What metrics are used to evaluate self-attention against recurrent and convolutional layers?",
        "question2": "Which three comparison factors are considered when analyzing self-attention versus RNNs and CNNs?",
        "question3": "How does the paper compare self-attention with recurrent and convolutional architectures?",
        "expected_chunk_ids": [14]
    },

    {
        "question1": "What findings indicate that self-attention may be easier to interpret?",
        "question2": "Why do the authors suggest self-attention improves model interpretability?",
        "question3": "What evidence supports the interpretability benefits of self-attention mechanisms?",
        "expected_chunk_ids": [15]
    },

    {
        "question1": "Which datasets and vocabulary sizes were used in the English-German and English-French translation benchmarks?",
        "question2": "What training corpora and vocabulary settings were employed for the Transformer translation experiments?",
        "question3": "What data sources and vocabulary sizes were chosen for the WMT English-German and English-French tasks?",
        "expected_chunk_ids": [15]
    },

    {
        "question1": "What optimization algorithm and learning-rate strategy were used to train the Transformer?",
        "question2": "How was the learning rate scheduled during Transformer training, and which optimizer was selected?",
        "question3": "What optimizer configuration and learning-rate schedule were employed in the original Transformer paper?",
        "expected_chunk_ids": [16]
    },

    {
        "question1": "What dropout setting and label smoothing parameter were used in the base Transformer model?",
        "question2": "Which dropout rate and label smoothing value were applied during training of the Transformer base configuration?",
        "question3": "What regularization parameters were chosen for the base Transformer, including dropout and label smoothing?",
        "expected_chunk_ids": [19]
    },

    {
        "question1": "What BLEU results were achieved by the Transformer on WMT 2014 English-to-German and English-to-French translation tasks?",
        "question2": "How well did the Transformer perform in terms of BLEU score on the WMT14 En-De and En-Fr benchmarks?",
        "question3": "What translation accuracy, measured by BLEU, was reported for the Transformer on English-German and English-French datasets?",
        "expected_chunk_ids": [20]
    }
]

evaluation_set = [
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

evaluation_set2 = [
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

evaluation_set_for_multiquery2 = [
    {
        "question1": "What is the name of the paper that first presented the Transformer model?",
        "question2": "Which research paper introduced the Transformer architecture to the machine learning community?",
        "question3": "What is the title of the publication in which the Transformer architecture was proposed?",
        "expected_chunk_ids": [0]
    },

    {
        "question1": "What fundamental mechanism enables the Transformer to operate without recurrence or convolution?",
        "question2": "Which innovation allows the Transformer to replace both RNNs and CNNs?",
        "question3": "What core idea makes it possible for the Transformer to eliminate recurrent and convolutional layers?",
        "expected_chunk_ids": [1]
    },

    {
        "question1": "Why are recurrent neural networks difficult to train in parallel?",
        "question2": "What characteristic of recurrent models limits parallel computation during training?",
        "question3": "How does the sequential processing nature of RNNs hinder parallelization?",
        "expected_chunk_ids": [2]
    },

    {
        "question1": "How does the Transformer handle long-range dependencies more effectively than ConvS2S and ByteNet?",
        "question2": "What advantage does self-attention provide over ConvS2S and ByteNet for distant token relationships?",
        "question3": "Why can the Transformer connect far-apart positions more efficiently than ConvS2S and ByteNet?",
        "expected_chunk_ids": [3]
    },

    {
        "question1": "What components make up the Transformer's encoder-decoder structure?",
        "question2": "Which building blocks are used throughout the Transformer encoder and decoder?",
        "question3": "What are the main architectural elements of the Transformer model?",
        "expected_chunk_ids": [4]
    },

    {
        "question1": "How many encoder layers are used in the Transformer and what does each layer contain?",
        "question2": "What is the structure of a Transformer encoder layer and how many such layers are stacked?",
        "question3": "Describe the encoder stack depth and the sub-layers included in each encoder block.",
        "expected_chunk_ids": [5]
    },

    {
        "question1": "In the Transformer, what inputs and outputs does an attention function relate?",
        "question2": "What does the attention mechanism transform or map between in the Transformer architecture?",
        "question3": "How is an attention function formally defined in terms of its inputs and outputs?",
        "expected_chunk_ids": [6]
    },

    {
        "question1": "How is Scaled Dot-Product Attention mathematically computed?",
        "question2": "What equation defines Scaled Dot-Product Attention in the Transformer?",
        "question3": "What is the mathematical expression used to calculate attention from queries, keys, and values?",
        "expected_chunk_ids": [7]
    },

    {
        "question1": "Why are queries, keys, and values projected into multiple subspaces in Multi-Head Attention?",
        "question2": "What benefit does using multiple learned projections provide in Multi-Head Attention?",
        "question3": "Why does the Transformer apply several separate projections to Q, K, and V vectors?",
        "expected_chunk_ids": [8]
    },

    {
        "question1": "Where are the three applications of multi-head attention found within the Transformer model?",
        "question2": "What are the different roles of multi-head attention in the encoder and decoder?",
        "question3": "How is multi-head attention utilized across the various parts of the Transformer architecture?",
        "expected_chunk_ids": [9]
    },

    {
        "question1": "Which nonlinear activation function is used in the Transformer's feed-forward network?",
        "question2": "What activation is applied between the two linear layers of the position-wise feed-forward network?",
        "question3": "Which activation function appears inside each Transformer feed-forward block?",
        "expected_chunk_ids": [10]
    },

    {
        "question1": "How does the Transformer share weights between embeddings and the output layer?",
        "question2": "In what way are embedding matrices reused for output projection in the Transformer?",
        "question3": "What weight-sharing strategy is used between the embedding and softmax layers?",
        "expected_chunk_ids": [11]
    },

    {
        "question1": "According to the complexity analysis, which layer achieves an O(1) maximum path length?",
        "question2": "What type of layer provides constant maximum path length in the comparison table?",
        "question3": "Which architecture has a shortest-path complexity of O(1) between positions?",
        "expected_chunk_ids": [12]
    },

    {
        "question1": "Why must positional information be added to Transformer inputs?",
        "question2": "What purpose do positional encodings serve in a self-attention-based model?",
        "question3": "Why are positional encodings required when using the Transformer architecture?",
        "expected_chunk_ids": [13]
    },

    {
        "question1": "Which metrics are used to compare self-attention with recurrent and convolutional layers?",
        "question2": "What three factors are considered when evaluating self-attention against RNNs and CNNs?",
        "question3": "How does the paper assess the advantages and disadvantages of self-attention compared to other layer types?",
        "expected_chunk_ids": [14]
    },

    {
        "question1": "What observations indicate that different attention heads specialize in distinct linguistic tasks?",
        "question2": "What evidence shows that separate attention heads learn different language-related functions?",
        "question3": "How do the authors demonstrate that attention heads capture different linguistic patterns?",
        "expected_chunk_ids": [15]
    },

    {
        "question1": "How many NVIDIA P100 GPUs were required to train the Transformer models?",
        "question2": "What hardware setup involving NVIDIA P100 GPUs was used during Transformer training?",
        "question3": "How many P100 GPUs did the authors use for model training experiments?",
        "expected_chunk_ids": [16]
    },

    {
        "question1": "Why is label smoothing applied when training Transformer models?",
        "question2": "What role does label smoothing play in the Transformer's training procedure?",
        "question3": "How does label smoothing contribute to the performance of the Transformer?",
        "expected_chunk_ids": [19]
    },

    {
        "question1": "What BLEU result was obtained by the Transformer big model on WMT14 English-to-German translation?",
        "question2": "How well did Transformer (big) perform on the WMT 2014 En-De benchmark in terms of BLEU score?",
        "question3": "What translation quality score did the large Transformer achieve on the English-German task?",
        "expected_chunk_ids": [20]
    },

    {
        "question1": "Why did the researchers conduct architecture variation studies on the Transformer?",
        "question2": "What was the motivation behind experimenting with different Transformer configurations?",
        "question3": "Why were model ablation and variation experiments performed in the Transformer paper?",
        "expected_chunk_ids": [21]
    },

    {
        "question1": "What BLEU score and model size were reported for the base Transformer in the variation experiments?",
        "question2": "How did the base Transformer perform and how many parameters did it contain?",
        "question3": "What translation accuracy and parameter count are associated with the baseline Transformer architecture?",
        "expected_chunk_ids": [22]
    },

    {
        "question1": "How did decreasing the key dimension dk affect Transformer performance?",
        "question2": "What impact on translation quality was observed when dk was reduced?",
        "question3": "What happened to model effectiveness after lowering the attention key dimension?",
        "expected_chunk_ids": [23]
    },

    {
        "question1": "How many WSJ training sentences from the Penn Treebank were used in the parsing experiments?",
        "question2": "What was the size of the Penn Treebank WSJ training set used for English constituency parsing?",
        "question3": "How many training examples from the WSJ corpus were employed in the parsing evaluation?",
        "expected_chunk_ids": [24]
    },

    {
        "question1": "What parsing performance did the four-layer Transformer achieve in the semi-supervised WSJ experiment?",
        "question2": "How accurate was the 4-layer Transformer on the WSJ constituency parsing benchmark?",
        "question3": "What parsing accuracy result was reported for the semi-supervised 4-layer Transformer model?",
        "expected_chunk_ids": [25]
    },

    {
        "question1": "What future directions for attention-based architectures are discussed in the conclusion?",
        "question2": "Which research opportunities do the authors identify for attention-only models?",
        "question3": "What extensions or future work do the authors propose for Transformer-style attention mechanisms?",
        "expected_chunk_ids": [27]
    }
]






final_results_per_question = []

for q_idx, item in enumerate(evaluation_set_for_multiquery, 1):

    print("\n" + "=" * 80)
    print(f"QUESTION {q_idx}")
    print("=" * 80)

    query_variations = [
        item["question1"],
        item["question2"],
        item["question3"]
    ]

    all_retrieval_results = []

    # ─────────────────────────────
    # 1. Multi-query retrieval
    # ─────────────────────────────
    for query in query_variations:
        docs = hybrid_retriever.invoke(query)
        all_retrieval_results.append(docs)

    # ─────────────────────────────
    # 2. RRF fusion
    # ─────────────────────────────
    fused_results = reciprocal_rank_fusion(
        all_retrieval_results,
        k=60,
        verbose=False
    )

    print(f"RRF produced {len(fused_results)} docs")

    # ─────────────────────────────
    # 3. Rerank (CRITICAL STEP)
    # ─────────────────────────────
    query_main = item["question1"]  # or original question
    fused_docs = [doc for doc, score in fused_results]

    reranked_results = rerank_with_retry(
        fused_docs,
        query_main
    )

    # ─────────────────────────────
    # 4. Store final output
    # ─────────────────────────────
    final_results_per_question.append({
        "question_index": q_idx,
        "reranked_docs": reranked_results,
        "expected_chunk_ids": item["expected_chunk_ids"]
    })

    # ─────────────────────────────
    # 5. Print final results
    # ─────────────────────────────
    print("\nFINAL RERANKED RESULTS:")
    for rank, doc in enumerate(reranked_results, 1):
        print(f"\nRank {rank}")
        print(doc.page_content[:200], "...\n")

print("\nDONE: Multi-query + RRF + Reranking pipeline complete")


def recall_at_k2(final_results_per_question, evaluation_set, k=3):

    hits = 0

    for i, item in enumerate(final_results_per_question):
        question = evaluation_set[i]["question"]
        expected_ids = set(item["expected_chunk_ids"])

        retrieved_docs = item["reranked_docs"]

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

def mean_reciprocal_rank2(final_results_per_question, evaluation_set):

    total_rr = 0

    for item in final_results_per_question:

        expected_ids = set(item["expected_chunk_ids"])

        docs = item["reranked_docs"]

        rr = 0

        for rank, doc in enumerate(docs, start=1):

            if doc.metadata["chunk_id"] in expected_ids:
                rr = 1 / rank
                break

        total_rr += rr

    return total_rr / len(evaluation_set)

recall = recall_at_k2( final_results_per_question, evaluation_set)
mrr = mean_reciprocal_rank2(final_results_per_question, evaluation_set)


print(
        f"Recall for mhr ={recall:.3f} "
        f"MRR for mhr ={mrr:.3f}"
    )





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






