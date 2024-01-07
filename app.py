# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(
    page_title='삶과 영혼의 비밀',
    layout='wide',
    page_icon=':robot_face:',
)

from scipy import spatial
import pandas as pd
import openai   # For calling the OpenAI API
import pickle
import base64
import io, os, re, random, datetime
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

def strings_ranked_by_relatedness(query, embeddings, relatedness_fn, top_n):
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in embeddings.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    return strings[:top_n], relatednesses[:top_n]

def query_message(query, embeddings, model):
    """Return a message for GPT, with relevant source texts pulled from embeddings."""
    if model.startswith('gpt-4-'):
        top_n = 64
    elif model.startswith('gpt-4'):
        top_n = 16
    elif model.startswith('gpt-3.5-turbo'):
        top_n = 32
    else:
        top_n = 8
    strings, relatednesses = strings_ranked_by_relatedness(
        query,
        embeddings,
        lambda subj1, subj2: 1 - spatial.distance.cosine(subj1, subj2),
        top_n,
    )

    message = f'다음 단서들을 사용하여 주어진 질문에 정확하게 답해주세요.\n\n\n===단서 시작===\n\n'
    for i, string in enumerate(strings):
        next_article = string.strip() + "\n"
        message += f"- {next_article}]\n\n"
    return i, message + "===단서 끝===\n\n"

def load_embeddings(name):
    source = "embeddings.csv"
    embeddings = pd.read_csv(source)
    embeddings.embedding = [eval(embedding) for embedding in embeddings.embedding]
    return embeddings

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
        st.sidebar.markdown("<p style='text-align: center;'>"+image_html+"</p>", unsafe_allow_html=True)
        st.sidebar.markdown("")

    init_chat(chat_state)
    
    # Set API key
    openai.organization = ""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Embeddings
    chat_state['embeddings'] = load_embeddings(squeeze_spaces(subject))

    # Generate a response
    def generate_response(query, is_first_attempt):
        embeddings = chat_state['embeddings']
        count, prompt = query_message(query, embeddings, model=GPT_MODEL)
        if is_first_attempt:
            chat_state['messages'].append({
                "role": "system",
                "content": f"당신은 {expertise}입니다."})
            extended_prompt = prompt + f"(최대한 답을 하려 노력하되, 도저히 답을 알 수 없는 경우 말을 지어내지 말고 '죄송합니다. 그 질문에 대한 답을 찾을 수 없습니다.' 라고 해주세요.)\n\nQUESTION: {query}"
            chat_state['messages'].append({
                "role": "user",
                "content": extended_prompt})
            # st.info(extended_prompt)
            print()
            print("===========================================")
            print(extended_prompt)
            print("===========================================")
            print()

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
        user_input = st.chat_input("무엇을 도와드릴까요?")
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
                    st.error("잠시 후에 다시 시도해 주세요.")

    # Bells and whistles
    with st.sidebar.expander("내보내기", expanded=False):
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
            file_name = st.text_input("파일명", squeeze_spaces(subject))
            if file_name:
                if file_type == "chat":
                    file_path = file_name + "_" + \
                        str(datetime.datetime.now())[5:19].replace(' ', '_') + ".chat"
                    pickled_ = pickle.dumps(dict(chat_state), pickle.HIGHEST_PROTOCOL)
                    st.download_button(label="확인", data=pickled_, file_name=file_path)
                else:       # "csv" or "html"
                    file_path = f"{file_name}.{file_type}"
                    download = st.download_button(
                        label="확인",
                        data=build_data(pd.DataFrame({
                            'Prompt': chat_state['prompt'],
                            'Response': chat_state['generated'],
                        })),
                        file_name=file_path,
                        mime=f'text/{file_type}')

    with st.sidebar.expander("불러오기", expanded=False):
        conversation = st.file_uploader('대화 파일 업로드', label_visibility='collapsed')
        if conversation and st.button("확인",
                key="ok_restore", help="이 메뉴를 실행하면 현재 진행중인 대화가 지워집니다!"):
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
GPT_MODEL = 'gpt-4'

expertise = 'Dr. Know-it-all'
temperature = 0.0

subject = '삶과 영혼의 비밀'
intro = "* 대승불교 양우회 발간 '삶과 영혼의 비밀'에 대한 질의응답 서비스입니다.<br/>* 책 내용과 다른 내용이 반환되는 경우도 있으니 참고용으로만 사용하시기 바랍니다."

###
# Launch the bot
interact()
