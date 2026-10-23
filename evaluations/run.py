"""Run a small deterministic retrieval, safety, citation, and stale-source evaluation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pipeline import ingest,retrieve,evidence_notices,answer_draft,validate_citations,redact_sensitive

docs,errors=ingest(ROOT/"knowledge");cases=json.loads((ROOT/"evaluations/cases.json").read_text());results=[]
for case in cases:
    role=case.get("role","agent"); found=retrieve(case["question"],docs,case.get("filters",{}),role); sources=[x["source"] for x in found]; notices=evidence_notices(found,"low" if not found else "medium"); passed=True; detail=[]
    if case.get("expect_source") and case["expect_source"] not in sources: passed=False;detail.append("expected source missing")
    if case.get("expect_type") and case.get("expect_source") in sources and next(x["type"] for x in found if x["source"]==case["expect_source"])!=case["expect_type"]: passed=False;detail.append("source kind mismatch")
    if case.get("expect_no_restricted") and any(x.get("roles")==["specialist","admin"] for x in found): passed=False;detail.append("restricted source leaked")
    if case.get("expect_notice")=="stale" and not any("stale" in n["text"].lower() for n in notices): passed=False;detail.append("stale notice missing")
    if case.get("expect_notice")=="insufficient" and found: passed=False;detail.append("unexpected evidence retrieved")
    if case.get("expect_redacted") and ("@" in redact_sensitive(case["question"]) or "4111" in redact_sensitive(case["question"])): passed=False;detail.append("sensitive data not redacted")
    if case.get("expect_no_restricted") and not found:
        draft=answer_draft(case["question"],found)
        if draft["confidence"]!="low" or "specialist" not in draft["reply"].lower():passed=False;detail.append("unsupported case did not escalate")
    results.append((case["id"],passed,detail))
passed=sum(ok for _,ok,_ in results)
print(f"Retrieval and safety cases: {passed}/{len(results)} passed")
for name,ok,detail in results: print(f"{'PASS' if ok else 'FAIL'} {name}"+(f" — {', '.join(detail)}" if detail else ""))
print("Citation claim validation: exercised with mocked model response in test_pipeline.py")
print("Groundedness against a live model: not run unless OPENAI_API_KEY and OPENAI_MODEL are configured.")
if errors: print("Indexing errors:","; ".join(errors))
sys.exit(0 if passed==len(results) and not errors else 1)
