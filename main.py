"""
build a research agent from scratch using lanchain
"""

# import keys from env
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# now the enviroment variables have all keys

import requests
from bs4 import BeautifulSoup

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# parse response to str
from langchain_core.output_parsers.string import StrOutputParser

from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda

# web search api
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# from langchain.utilities import DuckDuckGoSearchAPIWrapper

ddgo_search_wrapper = DuckDuckGoSearchAPIWrapper()

# number of results per question for web search scrapping
RESULT_PER_QUESTION = 1


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
    search_results = ddgo_search_wrapper.results(query, max_results=result_per_question)

    return [r["link"] for r in search_results]


if __name__ == "__main__":
    # print("Welcome to the research agent")
    # exit(0)

    # 1. get the web you want to question on
    # url = "https://docs.smith.langchain.com/how_to_guides"
    # page_content = scrape_text(url)[:10000]
    # print(page_content)

    # exit(0)
    # 2. get prompt web_search_template
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
    scrape_and_summarize_chain = (
        RunnablePassthrough.assign(
            # x:{'question': 'What is the best way to get started with langchain?', 'url': 'https://www.pluralsight.com/resources/blog/data/getting-started-langchain'}
            text=lambda x: scrape_text(x["url"])[
                :10000
            ]  # This will trigger scaping, more automated
        )  # take current input and pass into it, for this task we do web scraping
        | SUMMARY_PROMPT
        | ChatOpenAI(model="gpt-3.5-turbo-1106")
        | StrOutputParser()
    )

    final_chain = (
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

    # 4. execute the chain
    final_chain.invoke(
        {"question": "What is the best way to get started with langchain?"}
    )
