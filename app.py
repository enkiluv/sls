# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(
    page_title='삶과영혼의비밀',
    layout='wide',
    page_icon=':robot_face:',
)

import pandas as pd
from openai import OpenAI
import pickle, zipfile
import base64
import io, os, re, random, datetime
from scipy import spatial
from tavily import TavilyClient
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
    if 'openai_client' not in state:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            openai_key = st.sidebar.text_input("OpenAI API key", value="", type="password")
        if openai_key:
            state['openai_client'] = OpenAI(api_key=openai_key)
    

def clear_chat(state):
    def clear_one(key):
        del state[key][:]
    clear_one('prompt')
    clear_one('generated')

def compute_embedding(segment, model=EMBEDDING_MODEL):
    return st.session_state['openai_client'].embeddings.create(input=[segment], model=model).data[0].embedding

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
    return list(strings[:top_n]), list(relatednesses[:top_n])

def query_message(query, prevs, model, embeddings):
    strings = []
    relatednesses = []
    clues = ""
    
    strings, relatednesses = related_strings(
        query + prevs,
        embeddings,
        lambda subj1, subj2: 1 - spatial.distance.cosine(subj1, subj2),
        top_n
    )
    
    for string in strings:  # TODO: Must check if the clues are within the token-budget
        clues += string.strip() + "\n\n"
        
    if 'use_websearch' in chat_state and chat_state['use_websearch']:
        try:
            tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            web_clues = tavily_client.search(query=query + "(최신자료순)", search_depth="advanced", include_answer=True)

            # if 'answer' in web_clues:
            #     clue = web_clues['answer'].strip().replace('~', '-')
            #     strings.append(clue)
            #     relatednesses.append(0.5)
            #     clues += f"{clue}\n\n"
            if 'results' in web_clues:
                for i, result in enumerate(web_clues['results']):
                    clue = result['content'].strip()
                    strings.append(clue)
                    relatednesses.append(0.5)
                    clues += f"{clue}\n\n"
        except Exception as e:
            # st.exception(e)
            # st.error(str(e))
            pass

    chat_state.lookup = strings, relatednesses

    unknown_ = "죄송합니다. 정보가 부족하여 답을 알 수 없습니다."
    if not clues:
        return f"단서가 없으면 '{unknown_}' 라고 말해주세요.", []
    return f"""
다음 단서들을 최대한 활용하여 답하세요. 답변은 {expertise} 전문가의 용어나 문체를 적극 활용하고, 존댓말을 사용해주세요.

{clues}

""", strings
    
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
    
    if st.sidebar.checkbox("웹 검색"):
        chat_state['use_websearch'] = True
    else:
        chat_state['use_websearch'] = False
    
    # Embeddings
    if 'embeddings' not in chat_state:
        with st.spinner("색인 정보를 로드하는 중..."):
            chat_state['embeddings'] = load_embeddings()

    # Generate a response
    def generate_response(query):
        prevs = ""
        if len(chat_state.prompt) > 0:
            for i, _prompt in enumerate(chat_state.prompt):
                prevs += f"({_prompt}) "
        context = ""
        with st.spinner("단서 탐색 중..."):
            context, clues = query_message(query, prevs, model, chat_state.embeddings)
                
        if os.path.exists('addendum.txt'):
            with open('addendum.txt', 'r', encoding='utf-8') as file:
                content = file.read()
            context += content
   
        if prevs:
            context += f"\n\n이전에 {prevs}와 같은 질문들을 한 적이 있으니 아래 질문의 맥락을 파악하기 위해 참고만 하고 그 질문들에 대한 직접적인 답은 절대 하지 마세요.\n"
    
        context += f"\n\n[!!!! 이 질문에 답해주세요: {query} !!!!]"
            
        user_content = f"{context}\n\n특별한 지시가 없는 한 {language} 언어로 {expertise} 전문가인척 응답하되, {tone}하세요."
        user_content = re.sub(r'\n\s*\n', '\n\n', user_content)
        print(user_content)
    
        system_content = f"당신은 {tone}하는 {expertise} {language} 원어민 전문가입니다."
        
        with st.sidebar.expander("지식출처"):
            st.dataframe(pd.DataFrame({'참고정보': clues}), hide_index=True)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            stream = chat_state['openai_client'].chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                n=1,
                # top_p=1,
                stop=None,
                stream=True)
            for chunk in stream:
                cpart = chunk.choices[0].delta.content
                if cpart is not None:
                    full_response += cpart.replace('~', '-')
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
                    generated = generate_response(user_input)
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

model = 'gpt-4'
top_n = 10

expertise = '대승불교 양우종'
tone = '확실한 사실만 간략하게 언급'
temperature = 0.0
language = '한국어'

subject = '삶과 영혼의 비밀 + 생활 속의 대자유'
intro = "* 대승불교 양우회 발간 <span style='color: skyblue;'>삶과 영혼의 비밀</span>과 <span style='color: orange'>생활 속의 대자유</span>의 내용에 대한 질의응답 서비스입니다.<br/>* 제공된 정보가 정확하지 않을 수 있으니, 이 정보를 참고자료로만 사용하고 필요하면 직접 더 확인해 보세요."

###
# Launch the bot
interact()
