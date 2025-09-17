# general libraries
import pandas as pd
import numpy as np
import json
import traceback
import threading
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from IPython.display import Markdown
# pydantic
from pydantic import BaseModel, Field
from pydantic import TypeAdapter
# langchain libraries, adjust for LLM and orchestration framework used used
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

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from pydantic import TypeAdapter
from enum import Enum



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
    bm25_retrievers: Dict[str, Any] = Field(default_factory=dict)
    chroma_retriever: Any = None
    refine_counts: Dict[str, int] = Field(default_factory=dict)
    topic_keywords: Dict[str, list[str]] = Field(default_factory=dict)
    

# Setup LLMs
llm = ChatOllama(model="llama3.2")
llm_struct = ChatOllama(model='llama3.2').with_structured_output(Narrative, method='json_schema')
llm_struct_merge = ChatOllama(model='llama3.2').with_structured_output(MergedNarratives, method='json_schema')
llm_grader = ChatOllama(model='llama3.2').with_structured_output(GradedNarrative, method='json_schema')



# RETRIEVE
def retrieve_node(state: GraphState) -> GraphState:
    topic_id = state.topic_id
    keywords = state.topic_keywords[state.topic_id]
    query = " ".join(keywords)    

    if state.bm25_retrievers is None or state.chroma_retriever is None:
        raise ValueError("bm25_retrievers and chroma_retriever must be provided")
        
    docs_bm25 = state.bm25_retrievers[topic_id].invoke(query)
    docs_chroma = state.chroma_retriever.invoke(query)

    return GraphState(
        topic_id=topic_id,
        query=query,
        documents_bm25=docs_bm25,
        documents_chroma=docs_chroma,
        narratives=[]
    )

# EXTRACT

def is_blank_narrative(data):
    fields = ['actor', 'action', 'event', 'description']
    return all(not (data.get(field) or '').strip() for field in fields)

def extract_narrative(state: GraphState, max_attempts: int = 3) -> GraphState:
    topic_id = str(state.topic_id)
    documents_bm25 = state.documents_bm25 or []
    documents_chroma = state.documents_chroma or []
    attempt = 0
    last_error = None
    last_raw_result = None

    if state.pending_narratives_with_docs:
        updated_pending = {str(k): v for k, v in state.pending_narratives_with_docs.items()}
    else:
        updated_pending = {}

    while attempt < max_attempts:
        attempt += 1
        print(f"🔄 Extract attempt {attempt} for topic {topic_id}...")

        docs_text = "\n".join(get_page_content(doc) for doc in documents_bm25)
        combined_text = (f"""
        You are a information extraction system.
        Your task:
        From the following documents, extract ONLY the information present to fill the following JSON object:
        {{
          "actor": "",
          "action": "",
          "event": "",
          "description": ""
        }}
        Rules:
        - STRICTLY use only the information found in the provided documents.
        - Absolutely NO external knowledge, assumptions, or inferred details.
        - Your output will be discarded if it contains information not directly from the documents.
        - Do NOT copy or reuse the examples below.
        - "action" should include at least one verb.
        - "event" is the object of the action and can include nouns and noun phrases.
        - "actor" can be any entity or multiple entities (individual, group, institution, public entity, country, etc.).
        - ONLY if you cannot determine an "actor", use "user".
        - "description" must summarize the narrative in one sentence and must be consistent with "actor","action" and "event".
        - Output ONLY the JSON object, nothing else.


        DOCUMENTS:
        -------------------
        {docs_text}

        """  )


        #combined_text = "\n".join(get_page_content(doc) for doc in documents_bm25) + prompt
        result = llm_struct.invoke(combined_text)
        #combined_text = "\n".join(get_page_content(doc) for doc in documents_bm25)
        try:
            result = llm_struct.invoke(combined_text)
            print("RAW LLM result:", result)

            if isinstance(result, dict) and is_blank_narrative(result):
                print(f"⚠️ All narrative fields blank, retrying (attempt {attempt})...")
                continue

            if isinstance(result, Narrative):
                narrative = result
            elif isinstance(result, dict):
                narrative = Narrative.model_validate(result)
            else:
                narrative = Narrative.model_validate(result)

            print("Narrative before overwrite:", narrative)
            narrative.topic_id = topic_id
            print("Narrative after overwrite:", narrative)

            narrative_with_docs = ApprovedNarrativeWithDocs(
                narrative=narrative,
                documents_bm25=documents_bm25,
                documents_chroma=documents_chroma
            )

            updated_pending[topic_id] = narrative_with_docs

            return state.model_copy(update={
                "pending_narratives_with_docs": updated_pending
            })

        except ValidationError as ve:
            print(f"❌ Validation error parsing Narrative (attempt {attempt}): {ve}")
            print("Raw LLM output:", result)
            last_error = ve
            last_raw_result = result

        except Exception as e:
            print(f"❌ Unexpected error during extraction (attempt {attempt}): {e}")
            last_error = e
            last_raw_result = result

    print(f"❌ Failed to extract valid Narrative after {max_attempts} attempts. Last error: {last_error}")
    # Fallback: use model_construct to create a partial Narrative and pass it on
    if last_raw_result:
        # fill missing fields with empty strings
        if isinstance(last_raw_result, dict):
            narrative_data = {
                "topic_id": topic_id,
                "actor": last_raw_result.get("actor", ""),
                "action": last_raw_result.get("action", ""),
                "event": last_raw_result.get("event", ""),
                "description": last_raw_result.get("description", "")
            }
            narrative = Narrative.model_construct(**narrative_data)
        elif isinstance(last_raw_result, Narrative):
            narrative = last_raw_result
            narrative.topic_id = topic_id
        else:
            narrative = Narrative.model_construct(
                topic_id=topic_id, actor="", action="", event="", description=""
            )

        narrative_with_docs = ApprovedNarrativeWithDocs(
            narrative=narrative,
            documents_bm25=documents_bm25,
            documents_chroma=documents_chroma
        )
        updated_pending[topic_id] = narrative_with_docs
        return state.model_copy(update={
            "pending_narratives_with_docs": updated_pending
        })
    return state

    
# GRADE 
MAX_REFINES = 100

def auto_grade_if_incomplete(narrative: Narrative) -> Optional[GradedNarrative]:
    required_fields = ["actor", "action", "event", "description"]
    missing = [field for field in required_fields if not getattr(narrative, field, "").strip()]

    if missing:
        return GradedNarrative(
            grade=Grade.refine,
            explanation=f"Missing or empty fields: {', '.join(missing)}"
        )

    return None


def grade_narrative(state: GraphState) -> GraphState:
    if not state.pending_narratives_with_docs:
        print("⚠️ No pending narratives to grade.")
        return state.model_copy(update={
            "pending_narratives_with_docs": {},
            "grade_result": None
        })

    pending = state.pending_narratives_with_docs

    # Ensure topic_key is str for consistency
    topic_key = str(state.topic_id)
    if not isinstance(pending, dict) or topic_key not in pending:
        print(f"❌ Invalid or missing pending narrative for topic {topic_key}: {pending}")
        return state.model_copy(update={
            "pending_narratives_with_docs": pending if isinstance(pending, dict) else {},
            "grade_result": None
        })

    narrative_with_docs = pending[topic_key]
    narrative = narrative_with_docs.narrative

    # Log narrative fields, even if incomplete (for debugging)
    print("📝 Narrative to grade (may be partial):", narrative)

    docs_combined = (
        (narrative_with_docs.documents_chroma or []) +
        (narrative_with_docs.documents_bm25 or [])
    )

    context = "\n\n".join([get_page_content(doc) for doc in docs_combined])

    # Check for missing fields before calling the LLM
    graded = auto_grade_if_incomplete(narrative)
    if graded:
        print(f"⚠️ Narrative is incomplete. Auto-graded as 'refine': {graded.explanation}")
    else:
        try:
            prompt = f"""
You are a narrative fact-checker. Your task is to analyze a narrative in the context of supporting documents and determine if it is consistent.

### Rules for Grading

Start by assuming the narrative is **approved**. Change it to **refine** only if:

1. The narrative **contradicts** the context (i.e. directly conflicts).
2. The narrative includes hallucinations (i.e. facts not present in the context).

✅ Approve if:
- The narrative is CONSISTENT with the context.
- The narrative does not contradict the context (i.e. tells the opposite).
- Approximate matches exist (e.g. "America" ≈ "US").
- The actor is "user" (this is always valid and must be **approved** if other fields are valid).

🧠 Do NOT:
- Guess or invent information.
- Consider grammar, tone, or style.
- Penalize narratives that are vague but not contradictory.

Use the GradedNarrative schema with fields:
- grade: Either 'approved' or 'refine'
- explanation: A short explanation for the decision.

Context:
{context}

Narrative:
{narrative}
"""
            graded_raw = llm_grader.invoke(prompt)
            print("✅ Grading result:", graded_raw, flush=True)
            graded = (
                graded_raw if isinstance(graded_raw, GradedNarrative)
                else GradedNarrative.model_validate(graded_raw)
            )
        except Exception as e:
            print(f"⚠️ Could not parse grading result into GradedNarrative: {e}", flush=True)
            graded = None

    # Copy and prep state data
    approved_narratives = list(getattr(state, "approved_narratives", []))
    refine_counts = dict(getattr(state, "refine_counts", {}))
    pending_narratives_with_docs = dict(pending)
    refine_count = refine_counts.get(topic_key, 0)

    if graded and graded.grade == Grade.approved:
        approved_narratives.append(narrative_with_docs)
        refine_counts.pop(topic_key, None)
        pending_narratives_with_docs.pop(topic_key, None)

    elif graded and graded.grade == Grade.refine:
        refine_count += 1
        if refine_count >= MAX_REFINES:
            print(f"⚠️ Max refine attempts reached for topic {topic_key}, approving narrative.")
            approved_narratives.append(narrative_with_docs)
            refine_counts.pop(topic_key, None)
            pending_narratives_with_docs.pop(topic_key, None)
        else:
            refine_counts[topic_key] = refine_count

    else:
        # Invalid or missing grade — still increment refine count
        refine_count += 1
        print(f"⚠️ Grading failed or incomplete, incrementing refine_count: {refine_count} for topic {topic_key}")
        if refine_count >= MAX_REFINES:
            print(f"⚠️ Max refine attempts reached (fallback) for topic {topic_key}, approving narrative.")
            approved_narratives.append(narrative_with_docs)
            refine_counts.pop(topic_key, None)
            pending_narratives_with_docs.pop(topic_key, None)
        else:
            refine_counts[topic_key] = refine_count

    return state.model_copy(update={
        "approved_narratives": approved_narratives,
        "refine_counts": refine_counts,
        "grade_result": graded,
        "pending_narratives_with_docs": pending_narratives_with_docs
    })

def refine_narrative(state: GraphState) -> GraphState:
    topic_id = str(state.topic_id)
    documents_bm25 = state.documents_bm25 or []
    documents_chroma = state.documents_chroma or []

    # 🔍 Include grading explanation if the last grade was 'refine'
    explanation_text = ""
    if state.grade_result and state.grade_result.grade == Grade.refine:
        reason = state.grade_result.explanation.strip()
        if reason:
            explanation_text = (
                "Note: In the previous attempt, the narrative was marked for refinement because:\n"
                f"\"{reason}\"\n\n"
            )

    # 🧠 Construct prompt with explanation and extraction instructions
    docs_text = "\n".join(get_page_content(doc) for doc in documents_bm25)
    combined_text = (f"""
        You are a information extraction system.
        Your task:
        From the following documents, extract ONLY the information present to fill the following JSON object:
        {{
          "actor": "",
          "action": "",
          "event": "",
          "description": ""
        }}
        Rules:
        - STRICTLY use only the information found in the provided documents.
        - Absolutely NO external knowledge, assumptions, or inferred details.
        - Your output will be discarded if it contains information not directly from the documents.
        - Do NOT copy or reuse the examples below.
        - "action" should include at least one verb.
        - "event" is the object of the action and can include nouns and noun phrases.
        - "actor" can be any entity or multiple entities (individual, group, institution, public entity, country, etc.).
        - ONLY if you cannot determine an actor, use "user".
        - "description" must summarize the narrative in one sentence and must be consistent with "actor","action" and "event".
        - Output ONLY the JSON object, nothing else.


        DOCUMENTS:
        -------------------
        {docs_text}

        """  )




    

    last_raw_result = None
    try:
        result = llm_struct.invoke(combined_text)
        print("RAW LLM result (refine):", result)

        if isinstance(result, Narrative):
            narrative = result
        elif isinstance(result, dict):
            narrative = Narrative.model_validate(result)
        else:
            narrative = Narrative.model_validate(result)

        print("Narrative before overwrite:", narrative)
        narrative.topic_id = topic_id
        print("Narrative after overwrite:", narrative)

    except ValidationError as ve:
        print(f"❌ Validation error parsing Narrative (refine): {ve}")
        print("Raw LLM output:", result)
        last_raw_result = result
        if isinstance(last_raw_result, dict):
            narrative_data = {
                "topic_id": topic_id,
                "actor": last_raw_result.get("actor", ""),
                "action": last_raw_result.get("action", ""),
                "event": last_raw_result.get("event", ""),
                "description": last_raw_result.get("description", "")
            }
            narrative = Narrative.model_construct(**narrative_data)
        else:
            narrative = Narrative.model_construct(
                topic_id=topic_id, actor="", action="", event="", description=""
            )

    except Exception as e:
        print(f"❌ Unexpected error during refinement: {e}")
        narrative = Narrative.model_construct(
            topic_id=topic_id, actor="", action="", event="", description=""
        )

    narrative_with_docs = ApprovedNarrativeWithDocs(
        narrative=narrative,
        documents_bm25=documents_bm25,
        documents_chroma=documents_chroma
    )

    return state.model_copy(update={
        "pending_narratives_with_docs": {topic_id: narrative_with_docs}
    })

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("extract", extract_narrative)
    builder.add_node("refine", refine_narrative)
    builder.add_node("grade", grade_narrative)

    builder.set_entry_point("retrieve")

    # Linear flow
    builder.add_edge("retrieve", "extract")
    builder.add_edge("extract", "grade")
    builder.add_edge("refine", "grade")

    # Conditional routing after grading
    def route_after_grading(state: GraphState):
        grade_result = state.grade_result
        topic_key = str(state.topic_id)
        pending = state.pending_narratives_with_docs

        # If narrative is approved, or no longer pending (force-approved or otherwise), we're done
        if (grade_result and grade_result.grade == Grade.approved) or \
           (not pending or topic_key not in pending):
            print(f"🎉 Narrative approved or force-approved for topic {state.topic_id}.")
            return END

        # Otherwise, keep refining
        print(f"🔁 Refining narrative for topic {state.topic_id}...")
        return "refine"

    builder.add_conditional_edges("grade", route_after_grading, {
        "refine": "refine",
        END: END
    })

    return builder.compile()


def safe_model_dump(obj, _seen=None):
    """
    Recursively serialize a Pydantic model or nested structures,
    skipping non-serializable objects (like retrievers).
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return None  # prevent circular references
    _seen.add(obj_id)

    # Handle Pydantic BaseModel
    if isinstance(obj, BaseModel):
        return safe_model_dump(obj.model_dump(mode="python", by_alias=False), _seen)

    # Handle dicts
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            dumped = safe_model_dump(v, _seen)
            # Only include serializable items
            if dumped is not None and isinstance(dumped, (dict, list, str, int, float, bool)):
                result[k] = dumped
        return result

    # Handle lists
    elif isinstance(obj, list):
        return [safe_model_dump(i, _seen) for i in obj if i is not None]

    # Handle basic types
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # For anything else (like Chroma retriever), skip it
    else:
        return None



def convert_numpy_types(obj, _seen=None):
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    if isinstance(obj, dict):
        return {k: convert_numpy_types(v, _seen) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [convert_numpy_types(i, _seen) for i in obj if i is not None]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def get_page_content(doc):
    if hasattr(doc, "page_content"):
        return doc.page_content
    elif isinstance(doc, dict) and "page_content" in doc:
        return doc["page_content"]
    elif hasattr(doc, "get") and callable(doc.get):
        return doc.get("page_content", str(doc))
    else:
        return str(doc)

def run_narrative_extraction(topic_keywords: dict, output_dir: Path, bm25_retrievers: dict, chroma_retriever):
    from langgraph.graph import END
    
    topic_keywords = {str(k): v for k, v in topic_keywords.items()}
    output_dir.mkdir(exist_ok=True)
    
    graph = build_graph()
    all_approved_narratives = []
    topic_results = {}
    

    for topic_id, keywords in topic_keywords.items():
        try:
            print(f"\n🚀 Processing topic {topic_id}...")

            initial_state = GraphState(
                topic_id=topic_id,
                query=" ".join(keywords),
                pending_narratives_with_docs={},
                topic_keywords={str(topic_id): keywords},
                bm25_retrievers=bm25_retrievers,
                chroma_retriever=chroma_retriever
            )

            final_state = graph.invoke(initial_state, {"recursion_limit": 500})

            # Ensure final_state is a valid GraphState instance
            if not isinstance(final_state, GraphState):
                if isinstance(final_state, BaseModel):
                    raw_state = final_state.model_dump(mode="python", by_alias=False)
                elif isinstance(final_state, dict):
                    raw_state = final_state
                else:
                    raise TypeError(f"Unexpected type for final_state: {type(final_state)}")

                final_state = TypeAdapter(GraphState).validate_python(raw_state)


            # Step 3: Extract narratives
            approved_narratives = final_state.approved_narratives or []

            # Step 4: JSON-safe output (detect circular refs only at this stage)
            try:
                result_dict = safe_model_dump(final_state)        # Handles nested BaseModels
                result_dict = convert_numpy_types(result_dict)    # Handles NumPy types
                with open(output_dir / f"topic_{topic_id}.json", "w") as f:
                    json.dump(result_dict, f, indent=2)
            except Exception as serialization_error:
                print(f"⚠️ Failed to serialize topic {topic_id}: {serialization_error}")
                traceback.print_exc()



            topic_results[topic_id] = {
                "approved_narratives": approved_narratives,
                "final_state": final_state
            }
            all_approved_narratives.extend(approved_narratives)
            print(f"✅ Topic {topic_id} done. {len(approved_narratives)} narrative(s) approved.")

        except Exception as e:
            print(f"❌ Error processing topic {topic_id}: {e}")
            traceback.print_exc()

    # Save all approved narratives globally
    approved_path = output_dir / "approved_narratives_global.json"
    with open(approved_path, "w") as f:
        json.dump(convert_numpy_types([
            safe_model_dump(n) for n in all_approved_narratives
        ]), f, indent=2)
    print(f"📋 Saved {len(all_approved_narratives)} total approved narratives.")

    return all_approved_narratives, topic_results

