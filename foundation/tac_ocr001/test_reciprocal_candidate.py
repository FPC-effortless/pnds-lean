"""Exercise the same fail-closed pinned-source checks for the new OCR ID."""
import hashlib
import json
from foundation.langcache104 import test_reciprocal_candidate as inherited
from scripts.check_reciprocal_links import LEGACY_PATH,PEER_REGISTRY,RAW,ReciprocalError


class OCRCandidatePinTests(inherited.CandidatePinTests):
    def setUp(self):
        super().setUp()
        self.row["id"]="TAC-OCR-001-TAC"
        self.map["variant_id"]=self.row["id"]
        self.write(LEGACY_PATH,self.local)
        raw=self.write(self.row["source_map_path"],self.map)
        self.write(self.pin_path,{"schema":"pnds.tac_ocr001.tac_pin.v1","tac_metadata_commit":self.metadata,
            "tac_source_commit":self.source,"source_map_sha256":hashlib.sha256(raw).hexdigest()})
        self.remote[RAW+self.metadata+"/"+PEER_REGISTRY]=json.dumps({"prospective_candidates":[self.row]}).encode()
        self.remote[RAW+self.metadata+"/"+self.row["tac_source_map_path"]]=raw

    def test_unreviewed_ocr_id_still_blocks(self):
        self.row["id"]="TAC-OCR-999-UNREVIEWED"
        self.write(LEGACY_PATH,self.local)
        with self.assertRaisesRegex(ReciprocalError,"reviewed verifier"):
            self.check()
