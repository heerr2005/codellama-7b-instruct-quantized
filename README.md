# 🪄 CodeLlama-7B-Instruct Google Colab Chat

A production-ready Jupyter notebook for running CodeLlama-7B-Instruct on Google Colab with 8-bit quantization. This implementation provides a stable, memory-efficient way to interact with Meta's CodeLlama model for code generation and assistance.

## 🎯 Overview

This notebook demonstrates how to:
- Load and run CodeLlama-7B-Instruct efficiently on free Colab GPUs (T4/L4)
- Use 8-bit quantization to avoid out-of-memory crashes
- Create an interactive chat interface with conversation history
- Apply proper chat templates for optimal model performance

## 🚀 Quick Start

1. **Open in Google Colab**
   - Click the notebook link: `CodeLlama_7B_Colab_Chat.ipynb`
   - Ensure GPU runtime is enabled: `Runtime` → `Change runtime type` → `T4 GPU`

2. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - Wait for model download (first run only, ~13GB)

3. **Start Chatting**
   - Type your questions or coding requests
   - Type `exit` or `quit` to end the conversation

## 📋 Requirements

### Hardware
- **GPU**: T4 or L4 (available on free Colab tier)
- **RAM**: High-RAM runtime recommended
- **Storage**: ~13GB for model weights

### Software
- Python 3.10+
- transformers
- accelerate
- bitsandbytes
- torch

## 🏗️ Architecture

### Model Details
- **Name**: `codellama/CodeLlama-7b-Instruct-hf`
- **Size**: 7 billion parameters
- **Quantization**: 8-bit (reduces memory footprint by ~4x)
- **Device Mapping**: Automatic GPU/CPU distribution

### Why 8-bit Quantization?

The standard Hugging Face `pipeline()` API loads models in full precision (16-bit or 32-bit), which exceeds Colab's GPU memory limits and causes crashes. This notebook uses manual loading with 8-bit quantization via `bitsandbytes`, which:

✅ Reduces memory usage from ~28GB to ~7GB  
✅ Maintains model quality with minimal accuracy loss  
✅ Enables stable inference on free Colab GPUs  
✅ Follows production best practices

## 📖 Notebook Structure

### Step 1: Installation
Installs required Python packages: `transformers`, `accelerate`, and `bitsandbytes`.

### Step 2: Imports
Loads necessary Python modules for model operations.

### Step 3: Model Loading
```python
model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### Step 4: Sanity Test
Verifies model functionality with a simple prompt.

### Step 5: Chat Template
Formats conversation history using CodeLlama's official template structure.

### Step 6: Interactive Function
Implements a chat loop with:
- Persistent conversation history
- Proper role formatting (user/assistant)
- Configurable generation parameters

### Step 7: Launch Chat
Starts the interactive interface.

## 🎛️ Generation Parameters

Current configuration:
- `max_new_tokens`: 256 (adjustable for longer responses)
- `temperature`: 0.7 (controls randomness)
- `top_p`: 0.9 (nucleus sampling threshold)
- `do_sample`: True (enables sampling vs greedy decoding)

## 💡 Usage Examples

### Code Generation
```
You: Write a Python function to calculate fibonacci numbers
CodeLlama: [generates code with explanations]
```

### Code Explanation
```
You: Explain how binary search works
CodeLlama: [provides detailed explanation]
```

### Debugging Help
```
You: Why is my list comprehension giving an error?
CodeLlama: [analyzes and suggests fixes]
```

## ⚠️ Known Limitations

- **First Run**: Model download takes 5-10 minutes
- **Session Timeout**: Colab disconnects after inactivity (~90 minutes)
- **Context Window**: Limited to ~4096 tokens
- **Quantization Trade-off**: Slight quality reduction vs full precision

## 🔧 Customization

### Adjust Response Length
Change `max_new_tokens` in the generation call:
```python
max_new_tokens=512  # For longer responses
```

### Modify Temperature
Lower for more focused outputs, higher for creativity:
```python
temperature=0.5  # More deterministic
temperature=1.0  # More creative
```

### Add System Prompts
Prepend system instructions to the messages list:
```python
messages = [
    {"role": "system", "content": "You are an expert Python developer."},
    {"role": "user", "content": user_input}
]
```

## 🚀 Next Steps

Potential enhancements:
- **Web UI**: Integrate Gradio for browser-based interface
- **Chat Persistence**: Save conversation logs to Google Drive
- **Multi-turn Context**: Implement context window management
- **Fine-tuning**: Adapt model to specific coding styles
- **API Wrapper**: Deploy as REST API endpoint

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Error handling and validation
- Performance optimizations
- Additional utility functions
- Documentation enhancements

## 📄 License

This notebook uses:
- CodeLlama-7B-Instruct: [Llama 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE)
- Notebook code: MIT License (or specify your preferred license)

## 🔗 Resources

- [CodeLlama Model Card](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Meta AI CodeLlama Blog](https://ai.meta.com/blog/code-llama-large-language-model-coding/)

## 🐛 Troubleshooting

### Out of Memory Error
- Switch to High-RAM runtime
- Reduce `max_new_tokens`
- Clear outputs between runs

### Model Not Loading
- Check internet connection
- Verify GPU is enabled in runtime settings
- Try restarting runtime

### Slow Generation
- Expected behavior on T4 GPUs
- Consider L4 for 2-3x speedup (Colab Pro)

## ⭐ Acknowledgments

Built with:
- Meta's CodeLlama model
- Hugging Face Transformers library
- Google Colab infrastructure

---

**Note**: This is an educational project demonstrating LLM deployment techniques. For production use, consider proper API keys, authentication, and infrastructure.
