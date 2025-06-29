# Use a pipeline as a high-level helper
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model directly
pipe = pipeline("fill-mask", model="hfl/chinese-bert-wwm")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-bert-wwm")

model.save_pretrained("./chinese-bert-wwm")
tokenizer.save_pretrained("./chinese-bert-wwm")
