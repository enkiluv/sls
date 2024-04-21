# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(
    page_title='삶과영혼의비밀',
    layout='wide',
    page_icon=':robot_face:',
)

import pandas as pd
import openai   # For calling the OpenAI API
import pickle, zipfile
import base64
import io, os, re, random, datetime
from scipy import spatial
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

def load_embeddings(pkz="embeddings.pkz"):
    with open(pkz, 'rb') as f:
        zip_data = io.BytesIO(f.read())

    with zipfile.ZipFile(zip_data, 'r') as zf:
        first_file_name = zf.namelist()[0]
        with zf.open(first_file_name) as pkl_file:
            pickle_data = io.BytesIO(pkl_file.read())
            data = pickle.load(pickle_data)
    return data
    
# Set model and map model names to OpenAI model IDs
EMBEDDING_MODEL = "text-embedding-3-large"

# GPT_MODEL = "gpt-4-turbo"
# GPT_MODEL = "gpt-3.5-turbo"

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

def clear_chat(state):
    def clear_one(key):
        del state[key][:]
    clear_one('prompt')
    clear_one('generated')

def compute_embedding(segment, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=[segment], model=model)['data'][0]['embedding']

def related_strings(query, embeddings, relatedness_fn, top_n):
    query_embedding = compute_embedding(query)
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in embeddings.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    
    if not strings_and_relatednesses:
        return [], []  # 결과가 없을 경우 빈 리스트 반환
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(query, prevs, model, embeddings):
    """Return a message for GPT, with relevant source texts pulled from embeddings."""
    if model.startswith('gpt-4-'):
        top_n = 45
    elif model.startswith('gpt-4'):
        top_n = 15
    elif model.startswith('gpt-3.5-turbo'):
        top_n = 30
    else:
        top_n = 10

    strings, relatednesses = related_strings(
        query + prevs,
        embeddings,
        lambda subj1, subj2: 1 - spatial.distance.cosine(subj1, subj2),
        top_n
    )
    chat_state.lookup = strings, relatednesses

    clues = ""
    for string in strings:  # TODO: Must check if the clues are within the token-budget
        clues += string.strip() + "\n\n"
    if not clues:
        return "정보가 부족하여 답을 알 수 없습니다 라고 말해주세요.", []
    return f"""
** 다음 단서들 가운데 질문의 취지(의도)와 확실히 관련된 단서들만을 사용하여 absolutely factual 하게 답하세요. 단서들이 질문에서 언급된 핵심 단어나 개념을 포함하지 않으면 모르겠습니다 라고 짧게 말해주세요. 단서를 활용하여 답을 찾은 경우, 실제 응답에서 사용된 단서들에 대하여 응답 마지막에 bullet으로 2건 이내로 간략히 요약 정리해주세요. 답변은 {expertise} 전문가의 용어나 문체를 적극 사용하고, 공손한 말투를 사용해주세요. **

{clues}""", strings
    
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
    if 'embeddings' not in chat_state:
        with st.spinner("색인 정보를 로드하는 중..."):
            chat_state['embeddings'] = load_embeddings()

    # Generate a response
    def generate_response(query, is_first_attempt):
        prevs = ""
        if len(chat_state.prompt) > 0:
            for i, _prompt in enumerate(chat_state.prompt):
                prevs += f"({_prompt}) "
        context = ""
        with st.spinner("단서 탐색 중..."):
            context, clues = query_message(query, prevs, GPT_MODEL, chat_state.embeddings)
        
        if os.path.exists('addendum.txt'):
            with open('addendum.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            context += content
        
        if prevs:
            context += f"\n\n이전에 {prevs}와 같은 질문들을 한 적이 있으니 아래 질문의 의도와 맥락을 파악하기 위해 참고만 하고 그 질문들에 대한 답은 절대 하지 마세요.\n"

        context += f"\n\n[!!!! 이 질문에 답해주세요: {query} !!!!]"
        
        user_content = f"{context}\n\nUse Korean language if not stated elsewhere. Provide only factual answers. Pretend as if you were a {expertise} expert."
        user_content = re.sub(r'\n\s*\n', '\n\n', user_content)
        print(user_content)

        system_content = f"You are an absolutely factual {expertise} expert speaking Korean language natively."
        
        with st.sidebar.expander("지식출처"):
            st.dataframe(pd.DataFrame({'참고정보': clues}), hide_index=True)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=GPT_MODEL,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    n=1,
                    # top_p=1,
                    stop=None,
                    stream=True):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "_")
            new_answer = full_response.strip()
            message_placeholder.markdown(new_answer)
        return new_answer
        
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
                    error_msgs = str(e)
                    if "reduce the length of the messages" in error_msgs:
                        st.error(error_msgs)
                        break
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
            st.rerun()

###
GPT_MODEL = 'gpt-4'

expertise = '대승불교 양우종'
temperature = 0

subject = '삶과 영혼의 비밀 + 생활 속의 대자유'
intro = "* 대승불교 양우회 발간 <span style='color: skyblue;'>삶과 영혼의 비밀</span>과 <span style='color: orange'>생활 속의 대자유</span>의 내용에 대한 질의응답 서비스입니다.<br/>* 제공된 정보가 정확하지 않을 수 있으니, 이 정보를 참고자료로만 사용하고 필요하면 직접 더 확인해 보세요."

###
# Launch the bot
interact()
