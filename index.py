import os
from flask import Flask, request, jsonify
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# ================================================
# 1. Dados da Empresa (inseridos como string)
# ================================================
company_data = """
Nossa empresa XYZ tem a missão de fornecer produtos de alta qualidade com atendimento excepcional.
Oferecemos uma linha completa de kits de básicos e peças individuais.
Nossa política de trocas é flexível: se o produto apresentar defeitos ou não atender às expectativas,
o cliente pode solicitar a troca ou o reembolso dentro de 7 dias corridos após o recebimento.
O atendimento funciona de segunda a sexta, das 8h às 18h.
"""

# ================================================
# 2. Criação de um Dataset customizado
# ================================================
class CompanyDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=64):
        self.examples = []
        # Tokeniza o texto inteiro
        tokenized_text = tokenizer.encode(text)
        # Divide o texto em blocos de tamanho block_size
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(torch.tensor(tokenized_text[i:i+block_size]))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return {"input_ids": self.examples[i], "labels": self.examples[i]}

# ================================================
# 3. Fine-Tuning do Modelo GPT-2 com os Dados da Empresa
# ================================================
model_name = "gpt2"  # Você pode experimentar "distilgpt2" para um modelo menor
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cria o dataset a partir dos dados da empresa
dataset = CompanyDataset(company_data, tokenizer, block_size=64)

# Configura os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./company_model",
    overwrite_output_dir=True,
    num_train_epochs=1,                # Número de épocas – ajuste conforme sua necessidade
    per_device_train_batch_size=1,
    save_steps=10,
    logging_steps=5,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Se o modelo ajustado ainda não foi salvo, treina-o; caso contrário, carrega-o.
if not os.path.exists("company_model/pytorch_model.bin"):
    print("Iniciando o fine-tuning do modelo...")
    trainer.train()
    model.save_pretrained("company_model")
    tokenizer.save_pretrained("company_model")
    print("Fine-tuning concluído e modelo salvo.")
else:
    print("Carregando o modelo ajustado...")
    model = GPT2LMHeadModel.from_pretrained("company_model")
    tokenizer = GPT2Tokenizer.from_pretrained("company_model")
    print("Modelo carregado.")

# ================================================
# 4. Criação do Endpoint com Flask
# ================================================
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req_data = request.get_json()
        # Espera que o Dialogflow envie a pergunta em "queryResult.queryText"
        user_input = req_data.get("queryResult", {}).get("queryText", "")
        if not user_input:
            return jsonify({"fulfillmentText": "Nenhuma pergunta foi recebida."})
        
        # Gera uma resposta com o modelo ajustado
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        output_ids = model.generate(
            input_ids,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2
        )
        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return jsonify({"fulfillmentText": response_text})
    except Exception as e:
        return jsonify({"fulfillmentText": f"Erro interno no servidor: {e}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
