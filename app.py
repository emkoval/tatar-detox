import warnings
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", message=".*bitsandbytes.*")

app = FastAPI(
    title="Tatar Detoxifier API"
)

MODEL_PATH = "./gemma3-lora-tat-detox-new-prompt"

# Глобальные
model = None
tokenizer = None


def make_prompt(toxic_text: str) -> str:
    messages = [
        {"role": "system", "content": "Rewrite a toxic tatar phrase in non-toxic style. Save the wording, but detoxify it."},
        {"role": "user", "content": toxic_text.strip()}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if prompt.startswith("<bos>"):
        prompt = prompt[5:]
    return prompt


try:
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError("Токенизатор не содержит chat_template! Убедись, что tokenizer_config.json содержит `chat_template`.")

    print("Загрузка модели на CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,     # CPU → float32
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    print("Модель и токенизатор загружены. Chat template активен.")
except Exception as e:
    raise RuntimeError(f"Ошибка инициализации: {e}")


class TextRequest(BaseModel):
    text: str


class DetoxResponse(BaseModel):
    original: str
    detoxified: str


@app.post("/detoxify", response_model=DetoxResponse)
async def detoxify_text(request: TextRequest):
    toxic = request.text.strip()
    if not toxic:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    try:
        prompt = make_prompt(toxic)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        if "<start_of_turn>model\n" in full_response:
            detox = full_response.split("<start_of_turn>model\n")[-1].strip()
        else:
            detox = full_response[len(prompt):].strip() if full_response.startswith(prompt) else full_response.strip()

        detox = detox.split("<end_of_turn>")[0].split("<eos>")[0].strip()

        return DetoxResponse(original=toxic, detoxified=detox)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


@app.get("/health")
async def health_check():
    chat_tmpl = "ok" if tokenizer and hasattr(tokenizer, "chat_template") and tokenizer.chat_template else "no chat template"
    return {
        "status": "healthy",
        "device": str(next(model.parameters()).device),
        "chat_template": chat_tmpl,
        "example_prompt": make_prompt("Сәлам!"),
    }


@app.get("/")
async def root():
    return {
        "message": "Tatar Detox API",
    }