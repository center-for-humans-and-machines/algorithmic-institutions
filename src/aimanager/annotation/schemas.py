"""Pydantic v2 response models, one fixed schema per pass."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, create_model


class Quoted(BaseModel):
    description: str
    quote: str = Field(description="verbatim excerpt from the document")


# step 1a: free coding of a rule
class FreeCodes(BaseModel):
    mechanisms: List[Quoted]


# step 1b: notes record
class Reason(Quoted):
    pass


class Mechanism(Quoted):
    status: Literal["adopted", "rejected", "considered"]
    tested: bool
    source_version: Optional[str] = Field(
        description="earlier version the notes credit for this idea, else null"
    )
    reasons: List[Reason]


class TheoryMention(BaseModel):
    name: str
    quote: str


class NotesRecord(BaseModel):
    mechanisms: List[Mechanism]
    theory_mentions: List[TheoryMention]
    unlinked_findings: List[Quoted]


# clustering: proposal and refinement
class ProposedCode(BaseModel):
    name: str = Field(description="short snake_case identifier")
    label: str
    definition: str
    fits: List[str]
    does_not_fit: List[str]


class Proposal(BaseModel):
    codes: List[ProposedCode]


def assignment_model(code_names):
    """{code: one of the code names or "none"}; fixed for one codebook."""
    choices = tuple(code_names) + ("none",)
    return create_model("Assignment", code=(Literal[choices], ...))


class Judgement(BaseModel):
    present: bool
    quote: str


def judge_model(code_names):
    """One {present, quote} field per code, so every code gets an answer.

    Fields are c0, c1, ... aliased to the code names, so a code name can never
    clash with a pydantic attribute; read answers with judge_answers().
    """
    fields = {
        f"c{i}": (Judgement, Field(alias=name)) for i, name in enumerate(code_names)
    }
    return create_model("RuleJudgement", **fields)


def judge_answers(value, code_names):
    return {name: getattr(value, f"c{i}") for i, name in enumerate(code_names)}


def _inline(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        return {k: _inline(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def json_schema(model):
    """The model's JSON schema with $refs inlined (friendlier to guided decoding)."""
    schema = model.model_json_schema()
    return _inline(schema, schema.get("$defs", {}))
