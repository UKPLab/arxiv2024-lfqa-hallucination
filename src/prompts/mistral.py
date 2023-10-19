from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, BaseChatPromptTemplate, PromptTemplate
from pydantic import BaseModel
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate, FewShotPromptTemplate


SINGLE_TURN_PROMPT = """\
<s>[INST] {system_prompt} {user_prompt_1} [/INST]
"""


MULTI_TURN_PROMPT = """\
<s>[INST] {system_prompt} {user_prompt_1} [/INST] {assistant_response_1} </s><s>[INST] {user_prompt_2} [/INST]
"""


class MistralPromptTemplate(ChatPromptTemplate, BaseModel):
    pass


if __name__ == '__main__':
    params = {
        "system_message": "Act as a maths expert. Solve the question.",
        "few_shot_examples": [
            {'input': '3+4', 'output': '7'},
            {'input': '2+4', 'output': '6'}
        ],
        "prompt": "23+76"
    }
    from langchain.prompts import (
        FewShotChatMessagePromptTemplate,
        ChatPromptTemplate,
    )

    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]
    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    print(example_prompt)
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    print(few_shot_prompt.format())
