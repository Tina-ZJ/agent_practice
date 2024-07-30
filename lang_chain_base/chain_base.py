#!/usr/bin/env python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv(override=True)

def chain_example():

    # 1. Create prompt template
    system_template = "Translate the following into {language}:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system",system_template),("user","{text}")]
    )

    # 2. Create model
    model = ChatMistralAI()

    # 3. Create parser
    parser = StrOutputParser()

    # 4. create chain
    chain = prompt_template | model | parser

    result = chain.invoke({"language":"bengali","text":"I love you"})
    print(result)


if __name__=='__main__':
    chain_example()