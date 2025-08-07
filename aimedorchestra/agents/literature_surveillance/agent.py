# aimedorchestra/agents/literature_surveillance/agent.py
from sentence_transformers import SentenceTransformer, util
from transformers           import T5ForConditionalGeneration, T5Tokenizer
import torch, os, chardet, pathlib, tempfile, shutil

class LiteratureAgent:
    """
    Retrieval-augmented summariser for medical literature.
    * Auto-detects & fixes encoding problems by rewriting bad .txt files as UTF-8 *
    """
    def __init__(self, docs_folder: str = "data/literature", top_k_default: int = 3):
        # -------------------- models --------------------
        self.embedder  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model     = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # -------------------- load docs -----------------
        docs, names = [], []
        docs_path = pathlib.Path(docs_folder)
        if not docs_path.exists():
            raise FileNotFoundError(f"{docs_folder} does not exist.")

        for file in docs_path.glob("*.txt"):                 # only .txt
            try:  # first try UTF-8 fast-path
                text = file.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                # ---------- auto-fix encoding ----------
                raw_bytes = file.read_bytes()
                enc       = chardet.detect(raw_bytes)["encoding"] or "latin-1"
                text      = raw_bytes.decode(enc, errors="ignore").strip()

                # write back out as clean UTF-8
                with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
                    tmp.write(text)
                    tmp_path = pathlib.Path(tmp.name)
                shutil.move(tmp_path, file)                 # atomic replace
                print(f"[Encoding-fix] Re-saved {file.name} from {enc} → UTF-8")

            if text:
                docs.append(text)
                names.append(file.name)

        if not docs:
            raise ValueError(f"No valid text found in {docs_folder}")

        self.docs           = docs
        self.doc_names      = names
        self.doc_embeddings = self.embedder.encode(docs, convert_to_tensor=True)
        self.top_k_default  = top_k_default

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> str:
        k = min(top_k or self.top_k_default, len(self.docs))
        q_emb   = self.embedder.encode(query, convert_to_tensor=True)
        scores  = util.pytorch_cos_sim(q_emb, self.doc_embeddings)[0]
        top_ids = torch.topk(scores, k=k).indices.tolist()

        context = " ".join(self.docs[i] for i in top_ids)[:2000]  # truncate
        prompt  = "summarize: " + context
        inp_ids = self.tokenizer.encode(prompt, return_tensors="pt",
                                        max_length=512, truncation=True).to(self.device)

        summ_ids = self.model.generate(inp_ids, max_length=160, min_length=50,
                                       num_beams=4, length_penalty=2.0, early_stopping=True)
        summary  = self.tokenizer.decode(summ_ids[0], skip_special_tokens=True)
        cites    = "\n".join(f"- {self.doc_names[i]}" for i in top_ids)

        return f"Summary:\n{summary}\n\nCitations:\n{cites}"
