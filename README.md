# FacompBot

FacompBot é um chatbot interativo desenvolvido com Python e Streamlit, utilizando modelos de inteligência artificial do Google Generative AI para responder perguntas com base em documentos fornecidos.

## 📌 Funcionalidades
- Carrega arquivos de texto da pasta `train_files` e os transforma em embeddings.
- Armazena os embeddings em uma base vetorial FAISS para consultas eficientes.
- Utiliza o modelo `Gemini 1.5 Flash` para responder perguntas baseadas no contexto dos documentos.
- Interface interativa via Streamlit para entrada e saída de perguntas/respostas.

## 🚀 Como executar

### 1️⃣ Pré-requisitos
- Python 3.8+
- Criar e ativar um ambiente virtual:
  ```sh
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate  # Windows
  ```
- Instalar as dependências:
  ```sh
  pip install -r requirements.txt
  ```
- Criar um arquivo `.env` na raiz do projeto com a chave da API do Google:
  ```env
  GOOGLE_API_KEY=SUA_CHAVE_AQUI
  ```

### 2️⃣ Executar o chatbot
```sh
streamlit run facompbot.py
```

## 📁 Estrutura do Projeto
```
FacompBot/
│── train_files/        # Arquivos de texto usados para treinamento
│── facompbot.py        # Código principal do chatbot
│── requirements.txt    # Lista de dependências do projeto
│── .env                # Exemplo do arquivo de variáveis de ambiente
│── README.md           # Documentação do projeto
```

## 🛠 Tecnologias Utilizadas
- **Python**
- **Streamlit** (Interface gráfica)
- **LangChain** (Manipulação de dados e IA)
- **Google Generative AI** (Geração de respostas)
- **FAISS** (Armazenamento e busca eficiente de embeddings)

---
💡 *Projeto desenvolvido para facilitar o acesso à informação acadêmica de forma eficiente e automatizada.*

