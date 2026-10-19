import json
import unittest
from src.data.load_data import load_claims, load_wiki_sentences
from src.preprocessing.text_preprocessor import tokenize
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search
from src.retrieval.boolean_retriever import search as boolean_search
from src.evaluation.retrieval_metrics import precision_at_k, reciprocal_rank, evidence_set_complete
from src.verification.label_mapping import map_nli
from src.pipeline.fact_verifier import verify
from src.verification.feature_extractor import extract_features
from src.evaluation.fever_evaluation import acceptable_evidence_sets, fever_score
from scripts.run_retrieval_evaluation import evaluate as evaluate_retrieval

def make_index():
    idx = InvertedIndex(); idx.add({"id":"p0","page":"P","sentence_id":0,"text":"The Pacific Ocean is the largest ocean."})
    idx.add({"id":"p1","page":"Q","sentence_id":0,"text":"The Atlantic Ocean is smaller than the Pacific Ocean."})
    return idx

class CoreTests(unittest.TestCase):
 def test_preprocess_keeps_negation_numbers_and_unicode(self):
    self.assertEqual(tokenize("NOT 42 Café"), ["not", "42", "café"])

 def test_duplicate_and_invalid_records(self):
    idx = make_index()
    with self.assertRaises(ValueError): idx.add({"id":"p0","text":"duplicate"})
    with self.assertRaises(ValueError): idx.add({"id":"","text":"missing id"})

 def test_bm25_returns_ranked_metadata(self):
    rows = search("Pacific largest", make_index(), 2)
    self.assertEqual(rows[0]["id"], "p0"); self.assertGreater(rows[0]["score"], 0); self.assertEqual(rows[0]["rank"], 1)

 def test_boolean_and_and_not(self):
    idx = make_index()
    self.assertEqual(boolean_search("Pacific AND largest", idx), ["p0"])
    self.assertEqual(boolean_search("Pacific NOT largest", idx), ["p1"])

 def test_empty_unknown_and_verify_abstains(self):
    idx = make_index()
    self.assertEqual(search("", idx), []); self.assertEqual(search("zzzz", idx), [])
    self.assertEqual(verify("The moon is cheese", idx)["verdict"], "NOT ENOUGH INFO")

 def test_metrics_and_label_mapping(self):
    self.assertEqual(precision_at_k(["a","b"], {"a"}, 2), .5)
    self.assertEqual(reciprocal_rank(["b","a"], {"a"}), .5)
    self.assertTrue(evidence_set_complete({"a","b"}, [{"a","b"}]))
    self.assertEqual(map_nli(.8,.1), "SUPPORTS"); self.assertEqual(map_nli(.2,.1), "NOT ENOUGH INFO")

 def test_feature_extractor_is_interpretable_and_keeps_negation(self):
    features=extract_features("The moon is not 42", [{"text":"The moon is not 42","score":2.0}])
    self.assertEqual(len(features), 6); self.assertEqual(features[0], 1.0)
    self.assertEqual(features[2], 1.0); self.assertEqual(features[3], 1.0)

 def test_fever_annotation_groups_and_score(self):
    raw=[[[0,0,"Page_A",1],[0,1,"Page_B",2]],[[1,0,"Page_C",0]]]
    groups=acceptable_evidence_sets(raw)
    self.assertEqual(groups,[{"Page_A::1","Page_B::2"},{"Page_C::0"}])
    self.assertEqual(fever_score(["SUPPORTS","REFUTES","NOT ENOUGH INFO"],
                                 ["SUPPORTS","REFUTES","NOT ENOUGH INFO"],
                                 [True,False,True]),2/3)

 def test_retrieval_evaluation_checks_whole_acceptable_group(self):
    idx=InvertedIndex()
    idx.add({"id":"Page_A::1","page":"Page_A","sentence_id":1,"text":"A claim about the Pacific Ocean."})
    claims=[{"claim":"Pacific Ocean","evidence":[[[0,0,"Page_A",1]]]}]
    report=evaluate_retrieval(claims,idx,5)
    self.assertEqual(report["claims_with_evidence"],1)
    self.assertEqual(report["metrics"]["evidence_set_complete"],1.0)

 def test_jsonl_loader_handles_bad_line_and_duplicates(self):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        f=Path(directory)/"claims.jsonl"
        f.write_text(json.dumps({"id":1,"claim":"Claim","label":"SUPPORTS"})+"\n{bad}\n"+json.dumps({"id":1,"claim":"duplicate"}),encoding="utf-8")
        self.assertEqual(len(load_claims(f)), 1)

 def test_wiki_lines_are_flattened_and_keep_ids(self):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        f=Path(directory)/"wiki.jsonl"
        f.write_text(json.dumps({"title":"A Page","lines":"0\tFirst sentence.\n1\tSecond sentence."}),encoding="utf-8")
        rows=load_wiki_sentences(f)
        self.assertEqual([r["id"] for r in rows], ["A Page::0","A Page::1"])
