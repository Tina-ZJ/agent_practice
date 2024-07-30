from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv
load_dotenv(override=True)

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )


def classification():
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0) \
        .with_structured_output(Classification)

    chain = tagging_prompt | llm
    inp = "Weather is ok here, I can go outside without much more than a coat"
    result = chain.invoke({"input": inp})
    print(result)


if __name__=='__main__':
    classification()
