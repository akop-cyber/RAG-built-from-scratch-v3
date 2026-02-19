from loader import Loader
from chunker import Chunker
from embedder import Embedder
from vector import VectorStorage
from retriever import Retriever
from cleaner import clean_text
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
gc.collect()
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

text = Loader("nazi.pdf")
text_ = text.load()
chunk = Chunker()
chunks = chunk.chunker(text_)
#print(len(chunks))
#print(chunks[0][:200])

e = Embedder()
v = e.embed(chunks)

store = VectorStorage()
print(len(store))

store.add(v,chunks)
print(len(store))

ret = Retriever(store,e)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "unsloth/Llama-3.2-3B-Instruct"

token = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = config,
    device_map = "auto"
)

llm_engine = pipeline(
    "text-generation",
    model = model,
    tokenizer = token
)
print("Loading model....")

messages = [
    {
        "role": "system",
        "content": (
            "You are a versatile AI assistant. Your task is to provide a comprehensive answer "
            "using ONLY the provided context. Follow these rules:\n"
            "1. Identify and summarize all key points found in the text.\n"
            "2. Organize the information logically (e.g., bullet points).\n"
            "3. If the context does not contain the answer, state that clearly.\n"
            "4. Do not use outside knowledge.\n"

        )
    },
]


while True:
  query = input("YOU: ")
  if query == "/exit":
    break
  else:
    
    answer = ret.retrieve(query)
    combinedans = "\n\n".join(answer)
    messages.append({
        "role":"user",
        "content":f"Context:\n{combinedans}\n\nQuestion: {query}"
    }
    )

    for i, chunk in enumerate(answer, 1):
      cleaned = clean_text(chunk)
      print(f"{i}. {cleaned}\n")
 

    prompt = token.apply_chat_template(messages,tokenize = False,add_generation_prompt=True)
    output = llm_engine(prompt,max_new_tokens = 512,do_sample = False, temperature = 0.1)
    fa = output[0]["generated_text"].split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
  
    messages.append({
        "role":"assistant",
        "content":f"{fa}"
    })
    print(f"model says: {fa}")