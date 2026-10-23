import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pipeline

class FakeResponse:
    def __init__(self,payload): self.payload=payload
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self): return json.dumps(self.payload).encode()

class PipelineEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.docs,cls.errors=pipeline.ingest(ROOT/"knowledge")
    def test_all_sample_sources_parse_and_chunk(self):
        self.assertFalse(self.errors)
        self.assertGreaterEqual(len(self.docs),7)
    def test_product_docs_retrieved_for_export_howto(self):
        found=pipeline.retrieve("How do I export my Nimbus Cloud workspace?",self.docs,{"product":"Nimbus Cloud","issue_type":"Data export"},"agent")
        self.assertIn("nimbus-export-help.json",[d["source"] for d in found])
        self.assertEqual(found[0]["type"],"policy")
    def test_current_policy_outranks_resolved_ticket(self):
        found=pipeline.retrieve("When will my Nimbus Cloud export be ready?",self.docs,{"product":"Nimbus Cloud","issue_type":"Data export"},"specialist")
        self.assertEqual(found[0]["source"],"cloud-export-policy.json")
        self.assertTrue(all(d["type"]!="ticket" for d in found[:1]))
    def test_role_filter_hides_restricted_refund_policy(self):
        found=pipeline.retrieve("Can you approve this Atlas Billing refund?",self.docs,{"product":"Atlas Billing","issue_type":"Refund"},"agent")
        self.assertNotIn("atlas-refunds-policy.json",[d["source"] for d in found])
        self.assertNotIn("atlas-refund-legacy.json",[d["source"] for d in found])
    def test_stale_guidance_is_flagged(self):
        found=pipeline.retrieve("How do I export my Nimbus Cloud workspace?",self.docs,{"product":"Nimbus Cloud","issue_type":"Data export"},"agent")
        self.assertTrue(any("stale" in n["text"].lower() for n in pipeline.evidence_notices(found)))
    def test_conflicting_refund_policy_is_flagged_for_specialist(self):
        found=pipeline.retrieve("Can frontline support approve an Atlas Billing refund?",self.docs,{"product":"Atlas Billing","issue_type":"Refund"},"specialist")
        notices=pipeline.evidence_notices(found)
        self.assertTrue(any("conflicting" in n["text"].lower() for n in notices))
    def test_no_evidence_uses_escalation_reply(self):
        found=pipeline.retrieve("What is the galactic timezone for a lunar colony?",self.docs,{"product":"Nimbus Cloud"},"agent")
        answer=pipeline.answer_draft("What is the galactic timezone for a lunar colony?",found)
        self.assertEqual(answer["confidence"],"low")
        self.assertIn("specialist",answer["reply"].lower())
    def test_claims_need_valid_authoritative_citations(self):
        found=pipeline.retrieve("Where can I download my Atlas Billing invoice?",self.docs,{"product":"Atlas Billing","issue_type":"Billing"},"agent")
        cid=found[0]["chunk_id"]
        rows=[{"text":"Supported","citations":[cid]},{"text":"Uncited","citations":[]},{"text":"Made up","citations":["unknown"]}]
        self.assertEqual(pipeline.validate_citations(rows,found,authoritative_only=True),rows[:1])
    def test_india_upi_demo_uses_official_citation_and_safe_guidance(self):
        question="My UPI payment is pending and my bank account was debited. What should I do?"
        found=pipeline.retrieve(question,self.docs,{"product":"UPI payments","region":"India"},"agent")
        self.assertIn("upi-pending-payment.json",[d["source"] for d in found])
        with patch.dict(os.environ,{"OPENAI_API_KEY":"","OPENAI_MODEL":""}): draft=pipeline.answer_draft(question,found)
        self.assertIn("transaction history",draft["reply"].lower())
        self.assertIn("pin",draft["reply"].lower())
        self.assertEqual(draft["facts"][0]["citations"],[found[0]["chunk_id"]])
    def test_india_sample_answers_link_to_official_help(self):
        for question,product,filename,phrase in [
            ("How do I find the receipt for my Airtel prepaid recharge?","Airtel Prepaid","airtel-recharge-help.json","Download Receipt"),
            ("My Swiggy order has an issue. Where can I get help safely?","Swiggy","swiggy-order-support.json","Help section")]:
            with self.subTest(product=product):
                found=pipeline.retrieve(question,self.docs,{"product":product,"region":"India"},"agent")
                source=next(d for d in found if d["source"]==filename)
                self.assertTrue(source["source_url"].startswith("https://"))
                with patch.dict(os.environ,{"OPENAI_API_KEY":"","OPENAI_MODEL":""}): draft=pipeline.answer_draft(question,found)
                self.assertIn(phrase,draft["reply"])
                self.assertEqual(draft["facts"][0]["citations"],[source["chunk_id"]])
    def test_local_invoice_demo_draft_uses_source_citations(self):
        question="I’m our billing manager and need the October invoice PDF. Where can I download it?"
        found=pipeline.retrieve(question,self.docs,{"product":"Atlas Billing","issue_type":"Billing"},"agent")
        with patch.dict(os.environ,{"OPENAI_API_KEY":"","OPENAI_MODEL":""}): draft=pipeline.answer_draft(question,found)
        self.assertIn("Invoice history",draft["reply"])
        self.assertEqual(draft["confidence"],"high")
        self.assertEqual(draft["facts"][0]["citations"],[found[0]["chunk_id"]])
    def test_local_export_demo_does_not_promise_an_eta(self):
        question="When will my Nimbus Cloud export be ready?"
        found=pipeline.retrieve(question,self.docs,{"product":"Nimbus Cloud","issue_type":"Data export"},"agent")
        with patch.dict(os.environ,{"OPENAI_API_KEY":"","OPENAI_MODEL":""}): draft=pipeline.answer_draft(question,found)
        self.assertIn("completion time",draft["reply"])
        self.assertTrue(draft["facts"][0]["citations"])
    def test_generation_redacts_sensitive_data_and_discards_invalid_citations(self):
        found=pipeline.retrieve("How do I find the receipt for my Airtel prepaid recharge?",self.docs,{"product":"Airtel Prepaid","issue_type":"Recharge"},"agent")
        cid=found[0]["chunk_id"]
        response={"choices":[{"message":{"content":json.dumps({"reply":"Please check invoice history.","facts":[{"text":"Invoices are under Billing.","citations":[cid]},{"text":"Invented claim.","citations":["invalid"]}],"steps":[],"confidence":"high"})}}]}
        captured=[]
        def fake(req,timeout): captured.append(req.data.decode()); return FakeResponse(response)
        with patch.dict(os.environ,{"OPENAI_API_KEY":"test-only","OPENAI_MODEL":"test-model"}),patch("pipeline.urllib.request.urlopen",fake):
            result=pipeline.answer_draft("Contact me at person@example.com; my card is 4111 1111 1111 1111",found)
        self.assertNotIn("person@example.com",captured[0]); self.assertNotIn("4111 1111 1111 1111",captured[0])
        self.assertIn("[redacted email]",captured[0]); self.assertIn("[redacted phone or number]",captured[0])
        self.assertEqual(len(result["facts"]),1); self.assertEqual(result["facts"][0]["citations"],[cid])

if __name__=="__main__": unittest.main(verbosity=2)
