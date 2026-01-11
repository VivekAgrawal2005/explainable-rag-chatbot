"""
generate.py - Answer Generation using Ollama (LLaMA 3)

This module handles:
1. Calling Ollama API to generate answers using LLaMA 3
2. Restricting answers to retrieved context only
3. Handling cases where answer cannot be found

Dependencies:
- requests: For HTTP calls to Ollama API
- ollama: Ollama must be installed and running locally
"""

import json
from typing import List, Tuple


def check_ollama_available() -> bool:
    """
    Check if Ollama is installed and running.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        import requests
        
        # Try to connect to Ollama API (default: http://localhost:11434)
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
        
    except Exception:
        return False


def ensure_llama3_model() -> None:
    """
    Ensure LLaMA 3 model is available in Ollama.
    If not available, provide instructions to download it.
    """
    try:
        import requests
        
        # Check available models
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        
        # Check if llama3 is available
        model_names = [model.get('name', '') for model in models]
        llama3_models = [name for name in model_names if 'llama3' in name.lower()]
        
        if not llama3_models:
            print("\n" + "="*60)
            print("WARNING: LLaMA 3 model not found in Ollama!")
            print("="*60)
            print("To download LLaMA 3, run in terminal:")
            print("  ollama pull llama3")
            print("\nOr for a smaller model:")
            print("  ollama pull llama3:8b")
            print("="*60 + "\n")
        else:
            print(f"Found LLaMA 3 model(s): {', '.join(llama3_models)}")
            
    except Exception as e:
        print(f"Warning: Could not check Ollama models: {e}")


def generate_answer(
    query: str,
    context: str,
    model_name: str = "llama3"
) -> str:
    """
    Generate an answer using Ollama LLaMA 3 model.
    
    The prompt is designed to:
    1. Restrict the model to only use information from the provided context
    2. Return "Not found" if the answer cannot be determined from context
    3. Generate a clear, concise answer
    
    Args:
        query: User's question
        context: Retrieved text chunks (context for answering)
        model_name: Name of Ollama model to use (default: "llama3")
        
    Returns:
        Generated answer string
    """
    try:
        import requests
        
        # Construct prompt that restricts model to context only
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY information from the context above
- If the answer cannot be found in the context, respond with: "Not found in the provided documents."
- Be concise and accurate
- Do not make up information that is not in the context

Answer:"""

        # Prepare request to Ollama API
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }
        
        print(f"Generating answer using {model_name}...")
        
        # Send request to Ollama
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        answer = result.get('response', '').strip()
        
        # Handle empty or error responses
        if not answer:
            return "Not found in the provided documents."
        
        return answer
        
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running.\n"
            "Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be too slow or unavailable."
    except ImportError:
        raise ImportError("requests is required. Install with: pip install requests")
    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")


def generate_answer_with_fallback(
    query: str,
    context: str,
    model_name: str = "llama3"
) -> str:
    """
    Generate answer with automatic fallback to alternative models.
    
    If llama3 is not available, tries llama3:8b or llama3.1:8b.
    
    Args:
        query: User's question
        context: Retrieved text chunks
        model_name: Preferred model name
        
    Returns:
        Generated answer string
    """
    # List of models to try in order
    fallback_models = [model_name, "llama3:8b", "llama3.1:8b", "llama3.1"]
    
    last_error = None
    
    for model in fallback_models:
        try:
            return generate_answer(query, context, model)
        except Exception as e:
            last_error = e
            print(f"Failed with {model}, trying next model...")
            continue
    
    # If all models failed, return error message
    if last_error:
        return f"Error: Could not generate answer. {str(last_error)}"
    else:
        return "Not found in the provided documents."


# Example usage (for testing)
if __name__ == "__main__":
    print("Testing answer generation...")
    
    if check_ollama_available():
        ensure_llama3_model()
        
        test_query = "What is artificial intelligence?"
        test_context = "Artificial intelligence (AI) is the simulation of human intelligence by machines."
        
        try:
            answer = generate_answer(test_query, test_context)
            print(f"\nQuery: {test_query}")
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Ollama is not available. Please:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. Run: ollama pull llama3")
