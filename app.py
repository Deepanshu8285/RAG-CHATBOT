import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import base64
import time
import os
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


USER_AVATAR = "https://api.dicebear.com/7.x/personas/svg?seed=You"
BOT_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=Bot"
MAX_VISUALS = 25


# ---------- Storage paths (per user) ----------

def get_user_dir(username):
    path = f"user_data/{username}"
    os.makedirs(path, exist_ok=True)
    os.makedirs(f"{path}/visuals", exist_ok=True)
    return path


def get_faiss_path(username):
    return f"{get_user_dir(username)}/faiss_index"


# ---------- PDF extraction ----------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        pdf.seek(0)
    return text


def get_pdf_images(pdf_docs):
    images = []
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_bytes = pdf.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    continue

                if pil_image.width < 200 or pil_image.height < 200:
                    continue

                images.append({
                    "image": pil_image,
                    "source": pdf.name,
                    "page": page_num + 1,
                    "kind": "image"
                })
        pdf.seek(0)
    return images


def get_table_pages(pdf_docs):
    table_pages = []
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_bytes = pdf.read()

        pages_with_tables = set()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
            for page_num, page in enumerate(plumber_doc.pages):
                tables = page.find_tables()
                for table in tables:
                    rows = len(table.rows)
                    if rows >= 3:
                        pages_with_tables.add(page_num)
                        break

        if pages_with_tables:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in pages_with_tables:
                page = fitz_doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes))
                table_pages.append({
                    "image": pil_image,
                    "source": pdf.name,
                    "page": page_num + 1,
                    "kind": "table"
                })
        pdf.seek(0)
    return table_pages


def get_text_chunks(raw_text, source_name):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def describe_image(pil_image, llm, kind="image"):
    buffered = io.BytesIO()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    if kind == "table":
        instruction = (
            "This image shows a page containing one or more tables. "
            "Transcribe the table(s) accurately in a structured, readable format "
            "(preserve rows and columns clearly), and briefly note what the table represents."
        )
    else:
        instruction = (
            "Describe this image in detail, including any text, labels, charts, "
            "diagrams, or figures visible. Be specific and thorough, since this "
            "description will be used to answer questions about the image."
        )

    message = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }
    ])
    return message.content


# ---------- Vectorstore (with persistence + incremental add + citations) ----------

def build_or_update_vectorstore(existing_vectorstore, text_chunks, text_source, visual_items,
                                  llm, username, progress_callback=None):
    embeddings = OpenAIEmbeddings()

    all_texts = list(text_chunks)
    metadatas = [{"type": "text", "source": text_source, "page": None} for _ in text_chunks]

    visuals_dir = f"{get_user_dir(username)}/visuals"

    for i, item in enumerate(visual_items):
        if progress_callback:
            progress_callback(i + 1, len(visual_items))
        description = describe_image(item["image"], llm, kind=item["kind"])
        time.sleep(1)

        # Save the image to disk so it survives restarts
        safe_name = f"{item['kind']}_{item['source']}_{item['page']}_{i}.png".replace(" ", "_").replace("/", "_")
        visual_path = f"{visuals_dir}/{safe_name}"
        img_to_save = item["image"]
        if img_to_save.mode != "RGB":
            img_to_save = img_to_save.convert("RGB")
        img_to_save.save(visual_path, format="PNG")

        label = "Table" if item["kind"] == "table" else "Image"
        labeled_description = f"[{label} from {item['source']}, page {item['page']}]: {description}"
        all_texts.append(labeled_description)
        metadatas.append({
            "type": item["kind"],
            "source": item["source"],
            "page": item["page"],
            "visual_path": visual_path
        })

    if existing_vectorstore is not None:
        existing_vectorstore.add_texts(texts=all_texts, metadatas=metadatas)
        return existing_vectorstore
    else:
        return FAISS.from_texts(texts=all_texts, embedding=embeddings, metadatas=metadatas)


def load_vectorstore(username):
    path = get_faiss_path(username)
    if os.path.exists(path):
        embeddings = OpenAIEmbeddings()
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None


def save_vectorstore(vectorstore, username):
    vectorstore.save_local(get_faiss_path(username))


def get_doc_names(vectorstore):
    if vectorstore is None:
        return []
    sources = set()
    for doc in vectorstore.docstore._dict.values():
        src = doc.metadata.get("source")
        if src:
            sources.add(src)
    return sorted(sources)


def delete_document(vectorstore, source):
    ids_to_delete = [
        doc_id for doc_id, doc in vectorstore.docstore._dict.items()
        if doc.metadata.get("source") == source
    ]
    if ids_to_delete:
        vectorstore.delete(ids_to_delete)
    return vectorstore


# ---------- Answering (with citations + streaming) ----------

def get_answer_stream(vectorstore, question, chat_history):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build citation info
    citations = []
    relevant_visual_paths = []
    seen_citation_keys = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (src, page)
        if key not in seen_citation_keys:
            seen_citation_keys.add(key)
            if page:
                citations.append(f"{src} (page {page})")
            else:
                citations.append(f"{src}")
        if doc.metadata.get("type") in ("image", "table"):
            vp = doc.metadata.get("visual_path")
            if vp and os.path.exists(vp):
                relevant_visual_paths.append(vp)

    history_text = ""
    for entry in chat_history[-5:]:
        q, a = entry[0], entry[1]
        history_text += f"User: {q}\nAssistant: {a}\n"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""Use the following context and conversation history to answer the question.
The context may include descriptions of images or transcribed tables from the document; refer to them naturally in your answer.

Context: {context}
Conversation so far:
{history_text}

New question: {question}"""

    def token_generator():
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    return token_generator, citations, relevant_visual_paths


# ---------- Main app ----------

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚"
    )

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    if not st.session_state.get('authentication_status'):
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            authenticator.login(location='main')
        with tab2:
            try:
                email, username, name = authenticator.register_user(location='main')
                if email:
                    st.success('User registered successfully! Please log in.')
                    with open('config.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
            except Exception as e:
                st.error(e)

        if st.session_state.get('authentication_status') is False:
            st.error('Username/password is incorrect')
        return  # stop here until logged in

    # --- Logged in from this point on ---
    username = st.session_state['username']

    authenticator.logout('Logout', 'sidebar')
    st.sidebar.write(f"Signed in as **{st.session_state.get('name')}**")

    st.title("📚 Chat with Multiple PDFs")
    st.caption("Upload your PDFs (text + images + tables), then ask questions grounded in their content.")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore(username)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- Sidebar ---
    with st.sidebar:
        st.divider()
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            else:
                llm_vision = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                current_vs = st.session_state.vectorstore

                for pdf in pdf_docs:
                    with st.spinner(f"Reading {pdf.name}..."):
                        raw_text = get_pdf_text([pdf])
                        text_chunks = get_text_chunks(raw_text, pdf.name)

                    with st.spinner(f"Extracting images from {pdf.name}..."):
                        images = get_pdf_images([pdf])

                    with st.spinner(f"Detecting tables in {pdf.name}..."):
                        table_pages = get_table_pages([pdf])

                    visual_items = (images + table_pages)[:MAX_VISUALS]

                    if visual_items:
                        progress_bar = st.progress(0, text=f"Describing visuals in {pdf.name} (0/{len(visual_items)})...")

                        def update_progress(done, total):
                            progress_bar.progress(done / total, text=f"Describing visuals ({done}/{total})...")

                        current_vs = build_or_update_vectorstore(
                            current_vs, text_chunks, pdf.name, visual_items,
                            llm_vision, username, progress_callback=update_progress
                        )
                        progress_bar.empty()
                    else:
                        current_vs = build_or_update_vectorstore(
                            current_vs, text_chunks, pdf.name, [], llm_vision, username
                        )

                    st.success(f"{pdf.name}: found {len(images)} image(s), {len(table_pages)} table page(s).")

                st.session_state.vectorstore = current_vs
                save_vectorstore(current_vs, username)
                st.success("All documents processed and saved!")

        doc_names = get_doc_names(st.session_state.vectorstore)
        if doc_names:
            st.divider()
            st.markdown("**Loaded documents:**")
            for name in doc_names:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"- {name}")
                if col2.button("🗑️", key=f"del_{name}", help=f"Remove {name}"):
                    st.session_state.vectorstore = delete_document(st.session_state.vectorstore, name)
                    save_vectorstore(st.session_state.vectorstore, username)
                    st.rerun()

            if st.button("Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

            if st.session_state.chat_history:
                transcript = ""
                for entry in st.session_state.chat_history:
                    q, a = entry[0], entry[1]
                    transcript += f"You: {q}\n\nAssistant: {a}\n\n---\n\n"
                st.download_button(
                    "⬇️ Download conversation",
                    data=transcript,
                    file_name="conversation.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # --- Main chat area ---
    if not doc_names:
        st.info("👋 Upload one or more PDFs in the sidebar and click **Process** to get started.")
    else:
        for entry in st.session_state.chat_history:
            q, a = entry[0], entry[1]
            citations = entry[2] if len(entry) > 2 else []
            visual_paths = entry[3] if len(entry) > 3 else []

            with st.chat_message("user", avatar=USER_AVATAR):
                st.write(q)
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                st.write(a)
                for vp in visual_paths:
                    if os.path.exists(vp):
                        st.image(vp, width=350)
                if citations:
                    with st.expander("Sources"):
                        for c in citations:
                            st.markdown(f"- {c}")

        user_question = st.chat_input("Ask a question about your documents...")
        if user_question:
            with st.chat_message("user", avatar=USER_AVATAR):
                st.write(user_question)

            with st.chat_message("assistant", avatar=BOT_AVATAR):
                token_gen, citations, visual_paths = get_answer_stream(
                    st.session_state.vectorstore,
                    user_question,
                    st.session_state.chat_history
                )
                answer = st.write_stream(token_gen())

                for vp in visual_paths:
                    if os.path.exists(vp):
                        st.image(vp, width=350)
                if citations:
                    with st.expander("Sources"):
                        for c in citations:
                            st.markdown(f"- {c}")

            st.session_state.chat_history.append((user_question, answer, citations, visual_paths))
            st.rerun()

if __name__ == '__main__':
    main()