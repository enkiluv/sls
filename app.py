# -*- coding: cp949 -*-

import streamlit as st
st.set_page_config(
    page_title='��� ��ȥ�� ���',
    layout='wide',
    page_icon=':robot_face:',
)

import pandas as pd
import openai   # For calling the OpenAI API
import pickle
import base64
import io, os, re, random, datetime
from annoy import AnnoyIndex
from pathlib import Path
from PIL import Image

st.markdown("""<style>
    html, body, .stTextArea textarea {
        font-size: 14px;
    }
    section[data-testid="stSidebar"] {
        width: 200px !important;
        padding-top: 1rem;
    }
    div.row-widget.stRadio > div{flex-direction:row;}
</style>""", unsafe_allow_html=True)

def squeeze_spaces(s):
    s_without_spaces = re.sub(r'\s+', '', s)
    return s_without_spaces


# Set model and map model names to OpenAI model IDs
EMBEDDING_MODEL = "text-embedding-ada-002"

# GPT_MODEL = "gpt-4-1106-preview"
# GPT_MODEL = "gpt-3.5-turbo-16k"

MAX_RETRIES = 2
chat_state = st.session_state

logo_image = "logo_image.jpg"

# Initialize
def init_chat(state):
    def set_chat(key, value):
        if key not in state:
            state[key] = value
    set_chat('prompt', [])
    set_chat('generated', [])
    set_chat('messages', [])

def clear_chat(state):
    def clear_one(key):
        del state[key][:]
    clear_one('prompt')
    clear_one('generated')
    clear_one('messages')

def query_message(query, embeddings, model):
    """Return a message for GPT, with relevant source texts pulled from embeddings."""
    if model.startswith('gpt-4-'):
        top_n = 32
    elif model.startswith('gpt-4'):
        top_n = 8
    elif model.startswith('gpt-3.5-turbo'):
        top_n = 16
    else:
        top_n = 4

    def annoy_search(query_vector, annoy_index, texts, n_neighbors=5):
        nearest_ids, distances = annoy_index.get_nns_by_vector(
            query_vector, n_neighbors, include_distances=True)
        nearest_texts_distances = [(texts[i], distances[j]) for j, i in enumerate(nearest_ids)]
        return zip(*nearest_texts_distances)

    # annoy_index, texts = embeddings
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_vector = query_embedding_response["data"][0]["embedding"]
    strings, relatednesses = annoy_search(query_vector, *embeddings, top_n)

    message = f'���� �ܼ����� ����Ͽ� �־��� ������ ��Ȯ�ϰ� �����ּ���. ��� ������ {expertise}�� �մ��� �� ������ּ���. (��: "�Ƹ�"�� ���� �⵶���� ���� �������� ���� ��û���� ���� �� ���� ������� �ʾƾ� �մϴ�.)\n\n\n===�ܼ� ����===\n\n'
    for i, string in enumerate(strings):
        next_article = string.strip() + "\n"
        message += f"- {next_article}]\n\n"
    return i, message + "===�ܼ� ��===\n\n", strings

def load_embeddings(num_trees=10):
    import zipfile

    zip_file_path = "embeddings.zip"
    csv_file_path = "embeddings.csv"
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.csv'):
                    with z.open(filename) as csv_file:
                        df = pd.read_csv(csv_file)
    elif os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        raise Exception("CSV ���� �Ǵ� Zip ������ �������� �ʽ��ϴ�.")

    df.embedding = [eval(embedding) for embedding in df.embedding]

    texts = df['text'].tolist()     # �ؽ�Ʈ ������ ������ ����
    embeddings = df['embedding'].tolist()

    embedding_dim = len(embeddings[0])
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(num_trees)
    return annoy_index, texts       # Annoy �ε����� �ؽ�Ʈ ���� ��ȯ

def img_to_bytes(img_path):
    with Image.open(img_path) as img:
        original_width, original_height = img.size
        if original_width >= original_height:
            new_height = int((200 / original_width) * original_height)
            img_resized = img.resize((200, new_height))
        else:
            new_width = int((200 / original_height) * original_width)
            img_resized = img.resize((new_width, 200))
        img_bytes = io.BytesIO()
        img_resized.save(img_bytes, format=img.format)
        encoded = base64.b64encode(img_bytes.getvalue()).decode()
    return encoded

def interact():
    st.markdown(f"<h2 style='text-align: center;'><font color='green'>{subject}</font></h2>",
        unsafe_allow_html=True)

    if intro:
        st.markdown("")
        st.markdown(f"<p style='text-align: center;'>{intro}</p>", unsafe_allow_html=True)

    st.markdown("")

    def img_to_html(img_path):
        img_html = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}' style='display: block; margin-left: auto; margin-right: auto;'>"
        return img_html

    if logo_image and os.path.exists(logo_image):
        image_html = img_to_html(logo_image)
        st.sidebar.markdown(
            "<p style='text-align: center;'>"+image_html+"</p>", unsafe_allow_html=True)
        st.sidebar.markdown("")

    init_chat(chat_state)
    
    # Set API key
    openai.organization = ""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Embeddings
    chat_state['embeddings'] = load_embeddings()

    # Generate a response
    def generate_response(query, is_first_attempt):
        embeddings = chat_state['embeddings']
        count, prompt, clues = query_message(query, embeddings, model=GPT_MODEL)
        if is_first_attempt:
            chat_state['messages'].append({
                "role": "system",
                "content": f"����� {expertise} �������Դϴ�."})
            extended_prompt = prompt + f"""
������ ���� �� �� ���� ��� ���� ���� ����� ���� '�˼��մϴ�. �� ������ ���� �� �����ϴ�.' ��� ���ּ���. �ܼ��� Ȱ���Ͽ� ���� ã�� ���, ���� ���信�� ���� �ܼ��鿡 ���Ͽ� ���� �������� bullet���� �ִ��� ������ ��� �������ּ���.'

QUESTION: {query}"""
            chat_state['messages'].append({
                "role": "user",
                "content": extended_prompt})
            # st.info(extended_prompt)
            print()
            print("===========================================")
            print(extended_prompt)
            print("===========================================")
            print()

            with st.sidebar.expander("������ó"):
                st.dataframe(pd.DataFrame({'��������': clues}), hide_index=True)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=GPT_MODEL,
                    temperature=temperature,
                    messages=chat_state['messages'],
                    n=1,
                    stop=None,
                    stream=True):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "_")
            new_answer = full_response.strip()
            message_placeholder.markdown(new_answer)
        # Adjust messages to keep as minimal information as possible
        if len(chat_state['messages']) >= 2:
            del chat_state['messages'][-2]    # Remove the "system" role
            chat_state['messages'][-1] = {"role": "user", "content": query}
        # Keep the response for future references
        chat_state['messages'].append({"role": "assistant", "content": new_answer})
        return full_response

    def message_response(text):
        # st.success(text)
        st.markdown(f"<div style='background-color: #efefef; color: black; padding: 15px; font-size: 13px;'>{text}</div><br>", unsafe_allow_html=True)

    with st.container():
        for i in range(len(chat_state.generated)):
            st.chat_message("user").write(chat_state.prompt[i])
            st.chat_message("assistant").write(chat_state.generated[i])

        # A new query
        user_input = st.chat_input("������ ���͵帱���?")
        if user_input:
            st.chat_message("user").write(user_input)
            retries = 1
            while retries <= MAX_RETRIES:
                try:
                    generated = generate_response(user_input, retries==1)
                    break
                except Exception as e:
                    error_msgs = f"{str(e)}"
                    # st.error(error_msgs)
                    if "reduce the length of the messages" in error_msgs:
                        if len(chat_state['messages']) > 2:
                            count = random.randint(1, len(chat_state['messages']) - 2)
                            for _ in range(count):
                                del chat_state['messages'][0]     # LRU-principle
                        retries += 1
                        # continue
                    else:
                        retries = MAX_RETRIES + 2
                        break
            # After "while"...
            chat_state['prompt'].append(user_input.strip())
            if retries <= MAX_RETRIES:
                chat_state['generated'].append(generated)
            else:
                chat_state['generated'].append(error_msgs)
                if retries == MAX_RETRIES + 1:
                    st.error("��� �Ŀ� �ٽ� �õ��� �ּ���.")

    # Bells and whistles
    with st.sidebar.expander("��������", expanded=False):
        def to_csv(dataframe):
            csv_buffer = io.StringIO()
            dataframe.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return io.BytesIO(csv_buffer.getvalue().encode("cp949"))
    
        def to_html(dataframe):
            html = dataframe.to_html(index=False, escape=False).replace("\\n", '\0')
            return '<meta charset="utf-8">\n' + html

        with st.container():
            file_type = st.radio(label="Format",
                options=("As HTML", "As CSV", "As CHAT"), label_visibility="collapsed")
            file_type = file_type.split()[-1].lower()
            def build_data(chat):
                return (to_csv if file_type == "csv" else to_html)(chat)
            file_name = st.text_input("���ϸ�", squeeze_spaces(subject))
            if file_name:
                if file_type == "chat":
                    file_path = file_name + "_" + \
                        str(datetime.datetime.now())[5:19].replace(' ', '_') + ".chat"
                    pickled_ = pickle.dumps(dict(chat_state), pickle.HIGHEST_PROTOCOL)
                    st.download_button(label="Ȯ��", data=pickled_, file_name=file_path)
                else:       # "csv" or "html"
                    file_path = f"{file_name}.{file_type}"
                    download = st.download_button(
                        label="Ȯ��",
                        data=build_data(pd.DataFrame({
                            'Prompt': chat_state['prompt'],
                            'Response': chat_state['generated'],
                        })),
                        file_name=file_path,
                        mime=f'text/{file_type}')

    with st.sidebar.expander("�ҷ�����", expanded=False):
        conversation = st.file_uploader('��ȭ ���� ���ε�', label_visibility='collapsed')
        if conversation and st.button("Ȯ��",
                key="ok_restore", help="�� �޴��� �����ϸ� ���� �������� ��ȭ�� �������ϴ�!"):
            # Read the bytes of the file into a bytes object
            file_bytes = io.BytesIO(conversation.read())
            # Load the bytes object into a Python object using the pickle module
            saved_chat = pickle.load(file_bytes)
            clear_chat(chat_state)
            chat_state['prompt'] = saved_chat['prompt']
            chat_state['generated'] = saved_chat['generated']
            chat_state['messages'] = saved_chat['messages']
            st.rerun()

###
# GPT_MODEL = 'gpt-4'
GPT_MODEL = 'gpt-3.5-turbo'

expertise = '��ºұ� �����'
temperature = 0.2

subject = '��� ��ȥ�� ���'
intro = "* ��ºұ� ���ȸ �߰� <span style='color: skyblue;'>��� ��ȥ�� ���</span>�� <span style='color: orange'>��Ȱ ���� ������</span>�� ���뿡 ���� �������� �����Դϴ�.<br/>* ������ ������ ��Ȯ���� ���� �� ������, �� ������ �����ڷ�θ� ����ϰ� �ʿ��ϸ� ���� �� Ȯ���� ������."

###
# Launch the bot
interact()

