# RETRIEVE
def retrieve_node(state: GraphState) -> GraphState:
    topic_id = state.topic_id
    keywords = topic_keywords[str(topic_id)]
    query = " ".join(keywords)

    docs_bm25 = bm25_retrievers[topic_id].invoke(query)
    docs_chroma = chroma_retriever.invoke(query)

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

    
