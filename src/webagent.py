from openai import OpenAI
import os
import os
import sqlite3
from typing import Optional, Tuple
import time

import numpy as np


# init_db()

class WebAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        # Fallback client for embeddings using OpenRouter
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qa_cache.db")
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
        conn.commit()
        conn.close()


    def get_conn(self):
        return sqlite3.connect(self.db_path)


    def embed_text(self, text: str) -> np.ndarray:
        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            vec = resp.data[0].embedding
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            # If rate limit or quota error, use OpenRouter fallback
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                print(f"Primary embedding failed ({str(e)[:100]}), using OpenRouter fallback...")
                # Use OpenRouter with a model that supports embeddings
                # Note: OpenRouter doesn't directly support embeddings, so we'll use a simple hash-based approach
                # or return None to skip caching
                print("Warning: Embedding failed, skipping cache for this query")
                raise e
            else:
                raise


    def store_qa(self,question: str, answer: str):
        emb = self.embed_text(question)
        emb_bytes = emb.tobytes()

        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO qa_cache (question, answer, embedding) VALUES (?, ?, ?)",
            (question, answer, emb_bytes),
        )
        conn.commit()
        conn.close()

    
    def retrieve_qa(
        self,
        question: str,
        threshold: float = 0.8,
    ) -> Optional[Tuple[str, float]]:
        """
        Return cached answer and similarity if we find a close match.
        Otherwise return None.
        """
        conn = self.get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, question, answer, embedding FROM qa_cache")
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return None

        try:
            q_emb = self.embed_text(question)
        except Exception as e:
            # If embedding fails (e.g., rate limit), treat as cache miss
            print(f"Embedding failed during cache retrieval: {str(e)[:100]}")
            print("Falling back to exact string match...")
            # Try exact string match as fallback
            for row in rows:
                _id, q_text, ans_text, emb_blob = row
                if q_text.strip().lower() == question.strip().lower():
                    print("Found exact match in cache!")
                    return ans_text, 1.0
            return None
        
        q_norm = np.linalg.norm(q_emb)

        best_score = None
        best_answer = None

        for row in rows:
            _id, q_text, ans_text, emb_blob = row
            doc_emb = np.frombuffer(emb_blob, dtype=np.float32)
            d_norm = np.linalg.norm(doc_emb)
            if d_norm == 0 or q_norm == 0:
                continue

            score = float(np.dot(q_emb, doc_emb) / (q_norm * d_norm))

            if best_score is None or score > best_score:
                best_score = score
                best_answer = ans_text
        # print(f"Best similarity score from cache: {best_score:.3f}")
        if best_score is not None and best_score >= threshold:
            return best_answer, best_score

        return None


    def web_search_qa(self, question: str) -> str:
        try:
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=f"Explain all the vocabulary and key terms related to the field of electrical engineering in this question:\"{question}\"\n",
                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": "high",
                    }
                ],
            )

            texts = []
            for item in response.output:
                for part in item.content:
                    if part.type == "output_text":
                        texts.append(part.text)

            answer = "\n".join(texts)
            return answer
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                print(f"Azure OpenAI web search failed ({str(e)[:100]}), using OpenRouter fallback...")
                # Use OpenRouter chat completions as fallback (without web search)
                completion = self.openrouter_client.chat.completions.create(
                    model="openai/gpt-4o-2024-11-20",
                    temperature=0.2,
                    max_tokens=512,
                    messages=[
                        {"role": "system", "content": "You are an expert in electrical engineering. Explain technical terms and concepts clearly and concisely."},
                        {"role": "user", "content": f"Explain all the vocabulary and key terms related to the field of electrical engineering in this question:\"{question}\"\n"}
                    ]
                )
                return str(completion.choices[0].message.content)
            else:
                raise


    def answer_with_cache(self, question: str) -> str:
        try:
            cached = self.retrieve_qa(question)
            if cached is not None:
                answer, score = cached
                # print(f"Using cached answer from sqlite, similarity {score:.3f}")
                return answer
        except Exception as e:
            print(f"Cache retrieval failed: {str(e)[:100]}, proceeding without cache...")

        # No good match in sqlite cache, calling web search"
        answer = ""#self.web_search_qa(question)
        
        # try:
        #     self.store_qa(question, answer)
        # except Exception as e:
        #     print(f"Cache storage failed: {str(e)[:100]}, continuing anyway...")
        
        return answer
