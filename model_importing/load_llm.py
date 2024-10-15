from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class HuggingFaceLLM:
    def __init__(self, model_name, device="cuda", load_in_4bit=False, quantization_bits=None):
        """
        Initialize the Hugging Face LLM with options for GPU/CPU usage and quantization.
        
        Parameters:
        - model_name (str): The Hugging Face model to be loaded.
        - device (str): Device for model ('cuda' or 'cpu'). Default is 'cuda'.
        - load_in_4bit (bool): Whether to load the model with 4-bit quantization to save memory.
        - quantization_bits (int): Load the model with 8-bit or lower quantization.
        """
        self.model_name = model_name    
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the model with dynamic quantization if specified
        self.model = self._load_model(load_in_4bit, quantization_bits)
        
    def _load_model(self, load_in_4bit, quantization_bits):
        """
        Load the model with quantization or precision adjustments.
        
        Parameters:
        - load_in_4bit (bool): Enable 4-bit precision if True.
        - quantization_bits (int): Quantize model in 8-bit or lower precision.
        """
        model_kwargs = {"device_map": "auto" if self.device == "cuda" else None}

        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        elif quantization_bits:
            model_kwargs["load_in_8bit"] = True if quantization_bits == 8 else False

        try:
            return AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
    
    def generate_text(self, prompt, max_new_tokens=50, temperature=1.0, top_p=0.95, top_k=50):
        """
        Generate text from the model with options for controlling generation.
        
        Parameters:
        - prompt (str): The input text to generate from.
        - max_new_tokens (int): Maximum number of tokens to generate.
        - temperature (float): Controls randomness in generation.
        - top_p (float): Nucleus sampling probability.
        - top_k (int): The number of highest-probability vocabulary tokens to keep for sampling.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        try:
            generated_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k
            )
            return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

    def use_pipeline(self, task="text-generation"):
        """
        Optionally use Hugging Face pipeline for quick prototyping.
        
        Parameters:
        - task (str): The NLP task for the pipeline (default is text generation).
        """
        try:
            pipe = pipeline(task, model=self.model, tokenizer=self.tokenizer)
            return pipe
        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline: {e}")

    def save_model(self, path):
        """
        Save the model and tokenizer locally for later use.
        
        Parameters:
        - path (str): Directory path to save the model and tokenizer.
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise RuntimeError(f"Failed to save the model: {e}")

    def load_local_model(self, path):
        """
        Load a model and tokenizer from a local directory.
        
        Parameters:
        - path (str): Directory path where the model and tokenizer are stored.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load the local model: {e}")

# Example Usage
if __name__ == "__main__":
    model_name = "gpt2"  # You can use any Hugging Face model like 'mistralai/Mistral-7B-v0.1'
    llm = HuggingFaceLLM(model_name=model_name, load_in_4bit=True)
    
    # Text generation
    prompt = "The future of AI is"
    print(llm.generate_text(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9))
    
    # Optional pipeline usage
    text_pipeline = llm.use_pipeline()
    print(text_pipeline("Once upon a time in a faraway land"))
    
    # Save model locally
    llm.save_model("./local_model")
    
    # Load the model back from local storage
    llm.load_local_model("./local_model")
    print(llm.generate_text(prompt))
