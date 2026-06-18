from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeChunk:
    """知识库中可被独立检索的一段内容。"""

    source: str
    section: str
    text: str


@dataclass
class RetrievedEvidence:
    """一次 RAG 检索返回的证据及其相关度。"""

    source: str
    section: str
    text: str
    relevance_score: float
    retrieval_method: str


class RAGRetriever:
    """使用 OpenAI Embeddings 检索与异常最相关的业务知识。"""

    def __init__(
        self,
        knowledge_dir: str | Path = "knowledge/ymyc_anomaly",
        embedding_model: str | None = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model or os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievedEvidence]:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "OpenAI Embeddings are required for RAG retrieval."
            )

        # 第一步：把 Markdown 文件按照二级标题切成知识段落。
        chunks = self._load_chunks()
        if not chunks:
            return []

        documents = [
            f"{chunk.section} {chunk.text}"
            for chunk in chunks
        ]

        # 第二步：使用同一个 OpenAI embedding model 编码知识段落和查询。
        scores = self._openai_similarity_scores(
            documents=documents,
            query=query,
        )
        retrieval_method = f"openai:{self.embedding_model}"

        # 第三步：按相似度排序并返回最相关的知识段落。
        ranked_indexes = scores.argsort()[::-1][:top_k]

        return [
            RetrievedEvidence(
                source=chunks[index].source,
                section=chunks[index].section,
                text=chunks[index].text,
                relevance_score=round(float(scores[index]), 3),
                retrieval_method=retrieval_method,
            )
            for index in ranked_indexes
        ]

    def _openai_similarity_scores(
        self,
        documents: list[str],
        query: str,
    ) -> np.ndarray:
        """调用 OpenAI Embeddings，并计算查询与知识段落的余弦相似度。"""

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # 一次请求批量生成所有文档和查询的 embeddings，减少 API 调用次数。
        response = client.embeddings.create(
            model=self.embedding_model,
            input=[*documents, query],
            encoding_format="float",
        )
        embeddings = np.asarray(
            [item.embedding for item in response.data],
            dtype=float,
        )

        document_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1].reshape(1, -1)
        return cosine_similarity(
            query_embedding,
            document_embeddings,
        ).flatten()

    def _load_chunks(self) -> list[KnowledgeChunk]:
        """读取所有 Markdown，并按照 ## 标题切分。"""

        chunks = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            markdown = path.read_text(encoding="utf-8")
            chunks.extend(self._split_markdown(path.name, markdown))
        return chunks

    def _split_markdown(
        self,
        source: str,
        markdown: str,
    ) -> list[KnowledgeChunk]:
        """把一个 Markdown 文件拆成多个带标题的知识块。"""

        sections = re.split(r"^##\s+", markdown, flags=re.MULTILINE)
        chunks = []

        for section in sections[1:]:
            lines = section.strip().splitlines()
            if not lines:
                continue
            section_title = lines[0].strip()
            section_text = "\n".join(lines[1:]).strip()
            if section_text:
                chunks.append(
                    KnowledgeChunk(
                        source=source,
                        section=section_title,
                        text=section_text,
                    )
                )
        return chunks
