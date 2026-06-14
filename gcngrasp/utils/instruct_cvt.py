import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..api import Path
from ..api.utils import chat_tool
from ..dataset import task_grasp


class Args(BaseModel):
    cls: str = Field(
        description="The exact WordNet synset string from the candidate object list that best semantically matches the target object in the instruction."
    )
    task: str = Field(
        description="The exact task category string from the candidate task list that best semantically matches the action in the instruction."
    )


class InstructionConverter:
    classes = property(lambda self: self.database.classes)
    tasks = property(lambda self: self.database.tasks)

    def __init__(self,
                 model: str,
                 database: task_grasp.Database = None):
        if database is None:
            database = task_grasp.Database(Path("config/data.yaml").yaml())
        self.database = database
        self.in_prompt = Path("config/system/func_ins_cvt.md").read_text().format

        self.llm = ChatOpenAI(
            model=model,
            **chat_tool.CLIENT_KWARGS["bailian"]
        )
        self.llm_struct = self.llm.with_structured_output(Args)

    def __call__(self,
                 instruction: str,
                 image: np.ndarray = None) -> dict:
        print("Converting instruction...")
        content = [self.in_prompt(ins=instruction, classes=self.classes, tasks=self.tasks)]
        if image is not None: content.append(("image_url", image))

        content = chat_tool.make_content(content)
        response: BaseModel = self.llm_struct.invoke([HumanMessage(content)])
        return response.model_dump()
