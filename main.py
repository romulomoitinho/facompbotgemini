import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Função para ler e concatenar arquivos de texto da pasta 'train_files'
def obterArquivosTreinamento():
    texto = ""
    for arquivo in os.listdir("train_files"):
        # Verifica se o arquivo tem extensão .txt
        if arquivo.endswith(".txt"):
            with open(f"train_files/{arquivo}", 'r', encoding='utf-8') as leitorTexto:
                texto += leitorTexto.read()
    return texto

# Função para dividir o texto em trechos menores para facilitar o processamento
def dividirTextoEmTrechos(texto):
    divisorTexto = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    trechos = divisorTexto.split_text(texto)
    return trechos

# Função para criar e armazenar uma base vetorial FAISS com os embeddings do texto
def criarBaseVetorial(trechosTexto):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    baseVetorial = FAISS.from_texts(trechosTexto, embedding=embeddings)
    baseVetorial.save_local("faiss_index")

# Função para configurar e obter a cadeia conversacional baseada no modelo de IA
def obterCadeiaConversacional():
    modelo = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-001", temperature=0.5)
    
    # Template para estruturar as respostas do chatbot
    templatePrompt = """
    Responda a questão o mais detalhado possível a partir do contexto fornecido, certifique-se de fornecer todos os detalhes de forma que seja fácil a leitura, se a resposta não estiver no contexto fornecido, diga, "Infelizmente ainda não consigo respoder a pergunta. Para solucionar essa dúvida podes entrar em com a Representação dos Dicentes de Computação em recomp@gmail.com ou com a secretaria da faculdade de computação em facomp@ufpa.br".

    Contexto:
    {context}
    
    Pergunta:
    {question}
    
    Resposta:
    """
    
    prompt = PromptTemplate(template=templatePrompt, input_variables=["context", "question"])
    cadeia = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)
    
    return cadeia

# Função para processar a pergunta do usuário e buscar uma resposta baseada no contexto
def processarPerguntaUsuario(perguntaUsuario):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Carregar a base vetorial previamente criada
    baseVetorial = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Realizar busca semântica para encontrar os documentos mais relevantes
    documentosRelevantes = baseVetorial.similarity_search(perguntaUsuario, k=4)
    
    # Obter a cadeia conversacional configurada
    cadeia = obterCadeiaConversacional()
    
    # Obter a resposta a partir da cadeia conversacional
    resposta = cadeia({"input_documents": documentosRelevantes, "question": perguntaUsuario}, return_only_outputs=True)
    
    # Exibir a resposta na interface
    st.write("Resposta: ", resposta["output_text"])

# Executa a criação da base vetorial ao iniciar o script
textoBruto = obterArquivosTreinamento()
trechosTexto = dividirTextoEmTrechos(textoBruto)
criarBaseVetorial(trechosTexto)

# Função principal para iniciar a interface gráfica do chatbot
def main():
    st.set_page_config("FacompBot")
    st.header("Olá, eu sou o FacompBot! Faça uma pergunta.")
    
    # Campo de entrada para o usuário digitar sua pergunta
    perguntaUsuario = st.text_input("Digite sua pergunta:")
    
    if perguntaUsuario:
        processarPerguntaUsuario(perguntaUsuario)

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    main()