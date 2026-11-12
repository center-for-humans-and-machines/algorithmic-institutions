"""Pydantic v2 response models, one fixed schema per pass."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, create_model


class Quoted(BaseModel):
    description: str
    quote: str = Field(description="verbatim excerpt from the document")


# step 1a: free coding of a rule
# Generous list bounds so a runaway reply cannot spend max_tokens on one list.
class FreeCodes(BaseModel):
    mechanisms: List[Quoted] = Field(max_length=20)


# step 1b: notes record
class Reason(Quoted):
    stance: Literal["for", "against"] = Field(
        description="whether the reason argues for or against the mechanism"
    )
    evidence: Optional[str] = Field(
        description="what the reason rests on, in the agent's own terms taken from "
        "the notes; null if the notes do not say"
    )
    evidence_ref: Optional[str] = Field(
        description="what the evidence points to: an exploration number of these "
        "notes (e.g. 'exploration 3'), an earlier version's stem, 'none', or null "
        "if not stated"
    )


RELATIONS = ("new", "kept", "modified", "removed", "not_stated")


class Mechanism(Quoted):
    idea: str = Field(
        description="the mechanism stated positively as something a manager does, "
        "whether or not the notes adopt it"
    )
    status: Literal["adopted", "rejected", "considered"]
    tested: bool
    relation_to_previous: Literal[RELATIONS] = Field(
        description="how the notes present this mechanism relative to the previous "
        "version of the lineage"
    )
    source_version: Optional[str] = Field(
        description="earlier version the notes credit for this idea, else null"
    )
    reasons: List[Reason] = Field(max_length=12)


class TheoryMention(BaseModel):
    name: str
    kind: str = Field(description="e.g. concept, author, field, paper")
    quote: str


class NotesRecord(BaseModel):
    mechanisms: List[Mechanism] = Field(max_length=30)
    theory_mentions: List[TheoryMention] = Field(max_length=20)
    unlinked_findings: List[Quoted] = Field(max_length=30)


# clustering: proposal and refinement
class ProposedCode(BaseModel):
    name: str = Field(description="short snake_case identifier")
    label: str
    definition: str
    fits: List[str] = Field(max_length=4)
    does_not_fit: List[str] = Field(max_length=4)


class Proposal(BaseModel):
    codes: List[ProposedCode] = Field(max_length=60)


# answer values of an assignment that are not codes; no code may take these names
RESERVED = ("none", "failed")


def assignment_model(code_names):
    """{code: one of the code names or "none"}; fixed for one codebook."""
    clash = set(code_names) & set(RESERVED)
    if clash:
        raise ValueError(f"code names {sorted(clash)} are reserved")
    choices = tuple(code_names) + ("none",)
    return create_model("Assignment", code=(Literal[choices], ...))


# consolidation of a codebook: merges and families, over a fixed list of codes
def consolidation_model(code_names):
    """Merges of the given codes and a family for every surviving code.

    Code names are enums, so a reply can only name codes that exist.
    """
    Name = Literal[tuple(code_names)]
    Merge = create_model(
        "Merge",
        codes=(List[Name], Field(description="the codes that are one code")),
        keep=(Name, Field(description="the name that survives, one of `codes`")),
        label=(str, ...),
        definition=(str, Field(description="rewritten definition of the merged code")),
    )
    Family = create_model(
        "Family",
        name=(str, Field(description="short snake_case identifier")),
        definition=(str, Field(description="one sentence")),
        codes=(List[Name], ...),
    )
    return create_model(
        "Consolidation",
        merges=(List[Merge], Field(max_length=60)),
        families=(List[Family], Field(max_length=20)),
    )


class Judgement(BaseModel):
    # evidence first: the model has to find the line before it decides
    evidence: str = Field(
        description="the line of the rule that implements the code, copied "
        "verbatim; empty if the rule does not implement it"
    )
    present: bool
    switchable_off: bool = Field(
        description="true if a parameter value inside the declared sweep range "
        "disables the mechanism entirely"
    )


def judge_model(code_names):
    """One {evidence, present, switchable_off} field per code, in the given order.

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
