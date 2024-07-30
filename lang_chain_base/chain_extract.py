#!/usr/bin/env python
from typing import Optional,List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv(override=True)


system_prompt = (
    "You are an expert extraction algorithm. "
    "Only extract relevant information from the text. "
    "If you do not know the value of an attribute asked to extract, "
    "return null for the attribute's value."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{text}"),
    ]
)

class Person(BaseModel):
    """ Information about a person """
    name: Optional[str] = Field(default=None, description="The name")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known")
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

class Data(BaseModel):
    """Extracted data about people."""
    people: List[Person]


def extract():
    # set temperature=0 for high probability output
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=Data)
    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    result = runnable.invoke({"text": text})
    print(result)


if __name__=='__main__':
    extract()
