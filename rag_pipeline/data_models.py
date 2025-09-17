from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from pydantic import TypeAdapter
from enum import Enum

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# Setup pydantic models and graph state models
class Narrative(BaseModel):
    topic_id: str = Field(description="The topic ID of the narrative.")
    actor: str = Field(description="The actor(s) of the narrative.")
    action: str = Field(description="Action that is carried out by actor(s) or other entities or individuals.")
    event: str = Field(description="The event linking the actor(s) and their action.")
    description: str = Field(description="A one sentence long description of the narrative.")

class MergedNarratives(BaseModel):
    merged_narrative: Narrative
    merged_from: List[Narrative]

class Grade(str, Enum):
    approved = "approved"
    refine = "refine"

class GradedNarrative(BaseModel):
    grade: Grade  # Enum ensures only valid values
    explanation: str

class ApprovedNarrativeWithDocs(BaseModel):
    narrative: Narrative
    documents_bm25: List[Document]
    documents_chroma: List[Document]

class GraphState(BaseModel):
    topic_id: str = Field(description="The topic ID of the narrative.")
    query: Optional[str] = None
    documents_bm25: Optional[List[Document]] = None
    documents_chroma: Optional[List[Document]] = None
    narratives: Optional[List[Narrative]] = Field(default_factory=list)
    grade_result: Optional[GradedNarrative] = None
    approved_narratives: Optional[List[ApprovedNarrativeWithDocs]] = Field(default_factory=list)
    pending_narratives_with_docs: Dict[str, ApprovedNarrativeWithDocs] = Field(default_factory=dict)
    refine_counts: Dict[str, int] = Field(default_factory=dict)

# Setup LLMs
llm = ChatOllama(model="llama3.2")
llm_struct = ChatOllama(model='llama3.2').with_structured_output(Narrative, method='json_schema')
llm_struct_merge = ChatOllama(model='llama3.2').with_structured_output(MergedNarratives, method='json_schema')
llm_grader = ChatOllama(model='llama3.2').with_structured_output(GradedNarrative, method='json_schema')
