"""
build a research agent from scratch using lanchain
"""

from langchain_core.runnables.history import RunnableWithMessageHistory
import requests
from bs4 import BeautifulSoup

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# parse response to str
from langchain_core.output_parsers.string import StrOutputParser

from langchain_core.runnables.passthrough import RunnablePassthrough

# web search api
# error always appear rate limit
# chat memory
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    StreamlitChatMessageHistory,
)

from langchain_core.chat_history import BaseChatMessageHistory

# capture the response from the api as json
import json

# import openai for TTS
import openai
from openai import OpenAI

# TTS, STT
# from audiorecorder import audiorecorder

import streamlit as st
import os, time

key1 = st.text_input("Enter your OpenAI API key", type="password")

if key1:
    os.environ["OPENAI_API_KEY"] = key1
    st.write("API key set successfully!")
else:
    st.stop()


key2 = st.text_input("Enter your GoogleSerperAPIWrapper API key", type="password")

if key2:
    os.environ["SERPER_API_KEY"] = key2
    st.write("API key set successfully!")
else:
    st.stop()

from langchain_community.utilities import GoogleSerperAPIWrapper


# async TTS while output the response
# import asyncio

serper_searchwrapper = GoogleSerperAPIWrapper()

# number of results per question for web search scrapping
RESULT_PER_QUESTION = 3


# make a helper function to scrape the page
def scrape_text(url: str):
    # send a GET request to the url
    try:
        response = requests.get(url)

        # check if the request was successful
        if response.status_code == 200:
            # parse the html content
            soup = BeautifulSoup(response.text, "html.parser")

            # get all the text from the page
            text = soup.get_text(separator=" ", strip=True)

            # return the retrieved text
            return text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"

    except Exception as e:
        print(e)
        return f"An error occurred: {e}"


def search_web(query: str, result_per_question: int):
    """
    query: keywords to search for
    result_per_question: number of results to return per question

    @return: list of links
    """
    # print(f"query: {query}")
    # search the web for the query
    search_results = serper_searchwrapper.results(query)  # , result_per_question)
    search_results = search_results["organic"][:result_per_question]
    search_results = [result["link"] for result in search_results]

    return search_results


def flatten_2dlistofstr_2str(list_of_list):
    content = []
    for l in list_of_list:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


web_search_template = """{text}

--------------

Using the above text, answer in short the following question:

> {question}

--------------
if the question cannot be answered using the text, imply summarize the text. Include all factual
information, numbers, stats etc.
"""
SUMMARY_PROMPT = ChatPromptTemplate.from_template(web_search_template)
# print(SUMMARY_PROMPT)

# 3. create the chat chain
# chain includes:
# 0. passthrough: take the current input and pass into the chain
# 1. search: look up and retrieve information from duckduckgo, get most relevant 3,
# scrap it, pass into the llm
# 2. prompt: ask a question
# 3. chat: answer the question to str
scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(
        # x:{'question': 'What is the best way to get started with langchain?', 'url': 'https://www.pluralsight.com/resources/blog/data/getting-started-langchain'}
        text=lambda x: scrape_text(x["url"])[
            :10000
        ]  # This will trigger scaping, more automated
    )  # take current input and pass into it, for this task we do web scraping
    | SUMMARY_PROMPT
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
    # summary each element in the list
) | (
    # put each url and summary into a 2d list
    lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}"
)  # add the url to the summary

# return the actual web search result
web_search_chain = (
    # 1. get relevant urls
    RunnablePassthrough.assign(
        # return a number of list of urls
        # add a new key called url
        urls=lambda chain_params: search_web(
            chain_params["question"], RESULT_PER_QUESTION
        )
        # next stept turn urls dict to a list of urls
    )
    | (
        lambda chain_params: [
            {"question": chain_params["question"], "url": u}
            for u in chain_params["urls"]
        ]
    )  # 2. scrape urls
    | scrape_and_summarize_chain.map()  # 3. apply every element in the list
)  # "map": apply the chain to every element in the list

# 3. tune the search prompt
SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

# return list of questions
# temperature is the randomness of the response
search_question_chain = (
    SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads
)

# 4. execute the chain
# map the question list to dict so that the web_search_chain can parse
full_research_chain = (
    search_question_chain
    | (
        lambda x: [{"question": q} for q in x]
    )  # take the pervious chain output(list), put in a list of dict
    | web_search_chain.map()  # run each element in the list
)

# 5. we get the knowledge, now we need to write a report
WRITER_SYSTEM_PROMPT = """
You are an AI critical thinker research assistant. Your sole purpose 
is to write well written, critically acclaimed, objective and structured reports on given text.
"""

# NOTE: research prompt templates: https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py line 8 at def generate_report_prompt
REARCH_PROMPT = """Information:
----------------
{research_summary}
----------------


Using the above information, answer the following question or topic: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative,
in depth and comprehensive, with facts and numbers if available.

You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
You DO NOT write all used source urls at the end of the report as references, ONLY provided when user ask so and make sure to not add duplicated sources, but only one reference for each.
You MUST write the report in APA format.
You MUST limit the report no longer than 4096 characters(everything total).
Please do your best, this is very important to my career.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", WRITER_SYSTEM_PROMPT), ("user", REARCH_PROMPT)]
)

# full_research_chain will return list of list(2d list of knowledge per question)
# we need to conver the lists into a str
chain = (
    RunnablePassthrough.assign(
        research_summary=full_research_chain | flatten_2dlistofstr_2str
    )
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# add streamlit chabot history
streamlit_chat_history = StreamlitChatMessageHistory(key="chat_history")

# add the chat history to the chain
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: streamlit_chat_history,
    input_message_key="question",
    history_messages_key="chat_history",
)

# add history store at perticular session
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def stream_data(setntence: str):
    for word in setntence.split(" "):
        yield word + " "
        time.sleep(0.02)


# deal with data over 4096 characters
def stream_data_output(stream_data):
    for chunk in stream_data:
        yield chunk
        time.sleep(0.02)


if "messages" not in st.session_state:
    st.session_state.messages = []


# def stream_to_speakers(text: str) -> None:
#     """
#     give response to the user's input in audio
#
#     @param text: text to be converted to audio
#     """
#
#     start_time = time.time()
#
#     with openai.audio.speech.with_streaming_response.create(
#         model="tts-1",
#         voice="alloy",
#         response_format="mp3",
#         input=text,
#     ) as response:
#         print("saving response audio file  ...")
#         response.stream_to_file("response.mp3")
#
#     print(f"Done in {int((time.time() - start_time) * 1000)}ms.")


def main():
    st.title("Hello, I am a Research Assistant 🤖📚")
    st.subheader(
        "Ask me any question by typing into the chat box, and I will help you find the answer! 🤓"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # text input and audio input for question
    # audio record
    # audio_col, text_input_col = st.columns(2)
    question = None

    # with audio_col:
    #     audio = audiorecorder("Click to record", "Click to stop recording")
    #     # translate user audio to text question
    #     if len(audio) > 0:
    #         # To play audio in frontend:
    #         st.audio(audio.export().read())
    #
    #         # To save audio to a file, use pydub export method:
    #         audio.export("user_response.mp3", format="mp3")

    # with text_input_col:
    #     # question = st.chat_input("any questions on museum?")  # interact with openai
    #     question = st.chat_input("what do you want to know about museum?")
    #     st.session_state.messages.append({"role": "user", "content": question})
    #
    question = st.chat_input("what do you want to know about museum?")
    st.session_state.messages.append({"role": "user", "content": question})

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("question: ", question)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    if not question:
        print("trying audio...")
        # convert audio to text
        client = OpenAI()
        try:
            audio_file = open("user_response.mp3", "rb")
        except:
            print("I'm listening ...")
            st.stop()

        question = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

        # delete audio file after use
        os.remove("user_response.mp3")

        # only get str as input
        question = question.text

        with st.chat_message("human"):
            st.markdown(question)

        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("question: ", question)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    with st.chat_message("AI"):
        # streaming response
        with st.spinner("Thinking..."):
            # add user question, ai response to session history
            if question is not None:
                st.markdown(question)
                st.session_state.messages.append({"role": "user", "content": question})

            # TODO: chain invoke params
            ai_response = chain_with_memory.invoke(
                {"question": question, "chat_history": st.session_state.chat_history},
                config={"configurable": {"session_id": "2"}},
            )

            # convert response to audio
            # research report is TOO long to process audio, need limit to 4096 characters
            print("%%%%%%%%%%ai_response&&&&&&&&&&&&&&&&&&&&&&&")
            print(ai_response)
            print("%%%%%%%%%%ai_response&&&&&&&&&&&&&&&&&&&&&&&")

            if ai_response is not None:
                st.write_stream(stream_data_output(ai_response))
                st.session_state.messages.append({"role": "AI", "content": ai_response})

            # cannot do async to get text and audio same time, you need text generated before audio
            # tts_task = asyncio.create_task(stream_to_speakers(ai_response))
            # await tts_task

            # st.audio("response.mp3", format="audio/mp3", autoplay=True)

    # display chat history
    st.session_state.messages = list(
        filter(lambda x: x["content"] is not None, st.session_state.messages)
    )

    print("%%%%%%%%%%st.session_state.messages%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(st.session_state.messages)
    print("%%%%%%%%%%st.session_state.messages%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


if __name__ == "__main__":
    main()
