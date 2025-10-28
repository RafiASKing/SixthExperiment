Basis Pengetahuan LangChain & LangGraph untuk AI Copilot (Sintaks Terbaru)Dokumen ini adalah referensi sintaks yang padat dan akurat, dirancang khusus untuk AI Coding Assistants. Tujuannya adalah untuk menyediakan pola kode modern yang valid untuk LangChain dan LangGraph, dengan fokus utama pada LangChain Expression Language (LCEL) dan arsitektur stateful LangGraph.BAGIAN 1: LANGCHAIN CORE & LCELBagian ini mencakup paradigma fundamental dari LangChain modern: LangChain Expression Language (LCEL). Ini adalah dasar untuk membangun semua alur kerja.1.1. LCEL (LangChain Expression Language) - The Pipe |Operator | (pipe) adalah mekanisme deklaratif utama untuk merangkai komponen Runnable menjadi sebuah sekuens. Pendekatan ini memungkinkan LangChain untuk mengoptimalkan eksekusi runtime secara otomatis, termasuk dukungan untuk streaming, pemanggilan asinkron, dan paralelisasi.1SintaksPython# Operator `|` secara implisit membuat sebuah RunnableSequence.
# Output dari komponen di sebelah kiri menjadi input untuk komponen di sebelah kanan.
chain = runnable1 | runnable2 | runnable3
Contoh MinimalisPython# prompt (Runnable) -> model (Runnable) -> parser (Runnable)
chain = prompt | model | parser
1.2. Konfigurasi Runnable (.with_config)Metode .with_config() digunakan untuk melampirkan konfigurasi runtime ke pemanggilan Runnable (chain). Ini adalah cara standar untuk meneruskan parameter dinamis seperti ID sesi, metadata untuk pelacakan (tracing), atau callback kustom tanpa mengubah definisi chain itu sendiri.3SintaksPython# Memanggil chain dengan konfigurasi spesifik.
chain.invoke(input, config={"key": "value"})

# Membuat instance chain baru dengan konfigurasi yang terikat secara permanen.
configured_chain = chain.with_config(callbacks=my_callbacks)
Contoh Minimalis (dengan Callbacks)Pythonfrom langchain_core.callbacks import BaseCallbackHandler

# Definisikan sebuah callback handler kustom
class MyCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with inputs: {inputs}")

# Buat instance chain
# (Asumsikan 'prompt', 'model', dan 'parser' sudah didefinisikan)
chain = prompt | model | parser

# Buat instance chain baru dengan callback terikat
chain_with_callbacks = chain.with_config(callbacks=[MyCallbackHandler()])

# Panggil chain; callback akan otomatis terpicu
chain_with_callbacks.invoke({"input": "Tell me a fact."})
BAGIAN 2: KOMPONEN UTAMA (MODERN)Bagian ini merinci blok bangunan esensial untuk sebagian besar aplikasi LangChain, dengan penekanan pada path import modular yang modern.2.1. Models (Chat)Model bahasa diimpor dari paket integrasi spesifik (misalnya, langchain_google_genai, langchain_openai), bukan dari langchain atau langchain_community. Ini adalah bagian dari arsitektur modular LangChain.5Class: ChatGoogleGenerativeAIImport StatementPythonfrom langchain_google_genai import ChatGoogleGenerativeAI
Contoh InisialisasiPython# Pastikan environment variable GOOGLE_API_KEY sudah di-set
# gemini-2.5-flash adalah model yang cepat dan memiliki free tier.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
2.2. Prompt TemplatesChatPromptTemplate adalah standar untuk membuat prompt untuk model berbasis chat. Metode from_messages menyediakan cara yang paling ringkas dan mudah dibaca untuk mendefinisikan peran dan konten pesan.7Class: ChatPromptTemplateImport StatementPythonfrom langchain_core.prompts import ChatPromptTemplate
Contoh InisialisasiPython# Menggunakan daftar tuple (role, content)
prompt = ChatPromptTemplate.from_messages()
2.3. Output ParsersOutput parser mengubah output mentah dari model menjadi format yang lebih dapat digunakan. StrOutputParser dan JsonOutputParser adalah dua parser yang paling umum digunakan dan bersifat model-agnostik.8Class: StrOutputParserDeskripsi: Mengurai output model (biasanya AIMessage) menjadi string sederhana.Import StatementPythonfrom langchain_core.output_parsers import StrOutputParser
Contoh InisialisasiPythonparser = StrOutputParser()
Class: JsonOutputParserDeskripsi: Mengurai output model menjadi objek JSON. Ini bekerja dengan menyuntikkan instruksi format ke dalam prompt, membuatnya kompatibel dengan model LLM manapun yang mampu menghasilkan JSON.10Import StatementPythonfrom langchain_core.output_parsers import JsonOutputParser
Contoh InisialisasiPythonparser = JsonOutputParser()
2.4. EmbeddingsEmbeddings adalah komponen krusial dalam RAG yang mengubah teks menjadi representasi vektor numerik untuk pencarian kesamaan.Class: GoogleGenerativeAIEmbeddingsImport StatementPythonfrom langchain_google_genai import GoogleGenerativeAIEmbeddings
Contoh InisialisasiPython# Model embedding-001 adalah model yang efisien dan serbaguna.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
BAGIAN 3: CHAINS & MEMORY (MODERN)Bagian ini menunjukkan cara menggabungkan komponen-komponen di atas menjadi pola fungsional untuk interaksi stateless dan stateful.3.1. Membangun Chain SederhanaPola kanonis untuk chain sederhana adalah sekuens prompt | model | parser. Ini adalah fondasi dari hampir semua alur kerja LCEL.Pola Kode LengkapPythonfrom langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Inisialisasi komponen
prompt = ChatPromptTemplate.from_messages()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

# 2. Rangkai komponen menggunakan LCEL
chain = prompt | model | parser

# 3. Panggil chain dengan.invoke() untuk respons penuh
result = chain.invoke({"topic": "the moon"})
print(result)
Contoh Streaming dengan.stream()Salah satu keunggulan utama LCEL adalah dukungan streaming bawaan, yang memungkinkan output diterima secara bertahap.Python# Streaming response
for chunk in chain.stream({"topic": "the moon"}):
    print(chunk, end="", flush=True)
3.2. Chain dengan Memori PercakapanMemori percakapan diimplementasikan dengan membungkus chain stateless menggunakan RunnableWithMessageHistory. Pola ini memisahkan logika inti aplikasi dari manajemen state, memungkinkan backend memori (misalnya, in-memory, Redis, SQL) untuk ditukar dengan mudah.11Class Kunci: RunnableWithMessageHistoryImport StatementPythonfrom langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
Pola Kode LengkapPython# Asumsikan 'model' dan 'parser' sudah diinisialisasi dari contoh sebelumnya

# 1. Buat prompt yang menyertakan placeholder untuk riwayat pesan
prompt_with_history = ChatPromptTemplate.from_messages()

# 2. Buat chain inti
chain = prompt_with_history | model | parser

# 3. Buat penyimpanan memori (biasanya per sesi)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. Bungkus chain inti dengan RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. Panggil chain dengan config yang berisi session_id
config = {"configurable": {"session_id": "user123"}}
response = chain_with_history.invoke({"input": "Hi, I'm Bob!"}, config=config)
print(response)

# Panggil lagi, chain akan memiliki akses ke riwayat percakapan
response_2 = chain_with_history.invoke({"input": "What's my name?"}, config=config)
print(response_2)
BAGIAN 4: RAG (RETRIEVAL-AUGMENTED GENERATION)Bagian ini merinci pola LCEL paling penting untuk aplikasi RAG, yang menggabungkan pengambilan data dengan generasi bahasa.4.1. RetrieverSebuah retriever adalah Runnable yang mengambil dokumen sebagai respons terhadap sebuah query. Cara paling umum untuk membuatnya adalah dari instance vector store.SintaksPython# Asumsikan 'vectorstore' adalah instance yang sudah dikonfigurasi
# (misalnya, dari FAISS, Chroma, Pinecone)
retriever = vectorstore.as_retriever()
4.2. Merangkai Chain RAGPola RAG dalam LCEL memerlukan "percabangan" input awal (pertanyaan pengguna) ke dua jalur paralel: satu ke retriever untuk mendapatkan konteks, dan satu lagi untuk diteruskan ke prompt. RunnableParallel dan RunnablePassthrough adalah kunci untuk mencapai alur data ini.13Class Kunci: RunnablePassthrough, RunnableParallelImport StatementPythonfrom langchain_core.runnables import RunnablePassthrough, RunnableParallel
Pola Kode LengkapPython# Asumsikan 'retriever', 'model', dan 'parser' sudah diinisialisasi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 1. Buat prompt RAG yang membutuhkan 'context' dan 'question'
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 2. Buat setup untuk memproses input secara paralel
# - 'context' diisi oleh retriever
# - 'question' diteruskan dari input asli
setup = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)

# 3. Rangkai chain RAG lengkap
rag_chain = setup | prompt | model | parser

# 4. Panggil chain RAG
result = rag_chain.invoke("Where did Harrison work?")
print(result)
Catatan: setup juga dapat ditulis sebagai dictionary literal: {"context": retriever, "question": RunnablePassthrough()}.BAGIAN 5: LANGGRAPH - ALUR KERJA STATEFULLangGraph memperluas LCEL untuk membangun alur kerja yang stateful dan siklik (dengan loop), yang penting untuk agen. Konsep intinya adalah objek State yang eksplisit dan dapat dimodifikasi yang diedarkan di antara node-node dalam sebuah graf.155.1. StateState dari sebuah graf didefinisikan menggunakan TypedDict. Ini berfungsi sebagai skema untuk memori bersama dari graf, memastikan bahwa semua node membaca dan menulis ke struktur data yang konsisten.16Pola: Menggunakan TypedDictImport StatementPythonfrom typing import TypedDict, List
Contoh Definisi StatePythonfrom typing import TypedDict, List

# Mendefinisikan struktur state untuk graf.
# Setiap kunci adalah sebuah field dalam state bersama.
class MyGraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
5.2. Membangun GrafSetelah State didefinisikan, StateGraph digunakan untuk mendaftarkan node (fungsi) dan edge (transisi) untuk membangun alur kerja.Class Kunci: StateGraph, ENDImport StatementPythonfrom langgraph.graph import StateGraph, END
Pola Kode LengkapPythonfrom langgraph.graph import StateGraph, END

# Asumsikan ada fungsi 'retrieve' dan 'generate' yang menerima state
# dan mengembalikan pembaruan state dalam bentuk dictionary.
def retrieve(state):
    print("---NODE: RETRIEVE---")
    #... logika untuk mengambil dokumen
    return {"documents": ["doc1", "doc2"], "question": state["question"]}

def generate(state):
    print("---NODE: GENERATE---")
    #... logika untuk menghasilkan respons
    return {"generation": "This is the generated response."}

# 1. Definisikan alur kerja dengan StateGraph
workflow = StateGraph(MyGraphState)

# 2. Tambahkan node ke graf
workflow.add_node("retriever", retrieve)
workflow.add_node("generator", generate)

# 3. Tentukan titik masuk (entry point)
workflow.set_entry_point("retriever")

# 4. Tambahkan edge (penghubung antar node)
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END) # Mengakhiri graf setelah generator

# 5. Kompilasi graf menjadi objek yang dapat dijalankan
app = workflow.compile()

# 6. Panggil graf dengan input awal
inputs = {"question": "What is LangGraph?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Output from node '{key}': {value}")
