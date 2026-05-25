import asyncio
import logging
from typing import Any

import trafilatura
from langchain_core.tools import tool
from rerankers import Reranker

from app.models.response import Context
from app.pipeline.agent.tools.web_search.web_search_pipeline.chunker import Chunker
from app.pipeline.agent.tools.web_search.web_search_pipeline.link_deduplicator import (
    LinkDeduplicator,
)
from app.pipeline.agent.tools.web_search.web_search_pipeline.link_web_search import (
    LinkWebSearch,
)
from app.pipeline.agent.tools.web_search.web_search_pipeline.match_similar import (
    MatchSimilar,
)
from app.pipeline.agent.tools.web_search.web_search_pipeline.query_expander import (
    QueryExpander,
)
from app.pipeline.agent.tools.web_search.web_search_pipeline.rank import Rank

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


@tool  # type: ignore[misc]
async def SearchWeb(query: str) -> dict[str, Any]:
    """
    Search the web for current or factual information. Use this when the user asks about
    recent events, specific facts, or anything not covered by internal knowledge.
    """
    try:
        # 1. EXPAND QUERIES
        expander = QueryExpander()
        queries_string = expander.expanded_queries(query=query)

        # 2. PERFORM WEB SEARCH
        searcher = LinkWebSearch()
        all_work_data = await searcher.search_urls(
            queries_string=queries_string,
        )

        # 2.5 DEDUPLICATE THE LINKS
        deduplicator = LinkDeduplicator()
        list_urls = [
            str(link)
            for i in all_work_data
            if i and (link := i.get("link")) is not None
        ]
        urls_to_keep = deduplicator.deduplicate(list_urls)
        items_to_fetch = [d for d in all_work_data if d.get("link") in urls_to_keep]

        # 3. EXTRACT DATA
        async def fetch_page(data: dict[str, Any]) -> dict[str, Any] | None:
            url_link = data.get("link")
            if not url_link or not isinstance(url_link, str):
                return None
            try:
                raw_text = await searcher.fetch_url(url_link)
                if not raw_text:
                    logger.warning("Empty response for %s", url_link)
                    return None

                JUNK_PHRASES = {
                    "Log In",
                    "Sign Up",
                    "Please enable JavaScript",
                    "Enable JavaScript",
                }

                extracted = trafilatura.extract(raw_text) or ""
                is_junk = (
                    not extracted
                    or len(extracted) < 80
                    or any(p in extracted for p in JUNK_PHRASES)
                )
                clean_text = (
                    extracted
                    if not is_junk
                    else (data.get("body") or data.get("snippet") or "")
                )

                if clean_text:
                    logger.debug(
                        "OK extracted text from %s (%d chars)",
                        url_link,
                        len(clean_text),
                    )
                    return {
                        "title": data.get("title", ""),
                        "url": url_link,
                        "text": clean_text,
                    }
                else:
                    logger.warning("No text extractable for %s", url_link)

            except Exception as e:
                logger.error("Failed to fetch %s: %s", url_link, e)

            snippet = data.get("body", "").strip()
            if snippet:
                return {
                    "title": data.get("title", ""),
                    "url": url_link,
                    "text": snippet,
                }
            return None

        page_results = await asyncio.gather(*[fetch_page(d) for d in items_to_fetch])
        context_data = [r for r in page_results if r is not None]

        logger.debug(
            "Page results: %s",
            [
                (r.get("url"), len(r.get("text", ""))) if r else None
                for r in page_results
            ],
        )
        logger.debug("Context data count: %d", len(context_data))

        if not context_data:
            return {"error": "No page content could be extracted"}

        print(f"ALL THE CONTEXT DATA HAS BEEN GENERATED {context_data}")

        context_list = [
            Context(
                title=doc["title"],
                url=doc["url"],
                text=doc["text"],
            )
            for doc in context_data
        ]
        print(f"HERE'S THE FINAL PRODUCED CONTEXT_LIST: {context_list}")

        # 4 CHUNK THE DATA
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        chunked_docs = chunker.chunk_documents(context_list)

        # 5. BM25 SEARCH & RERANKING
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        ranker = Rank()
        match_maker = MatchSimilar(chunked_docs)
        seen_texts = set()
        final_docs = []

        for sub_query in queries_string:
            bm25_results = match_maker.match_similar_docs(sub_query)

            if not bm25_results:
                continue

            reranked = ranker.reranked_docs(
                reranker=reranker,
                initial_docs=bm25_results,
                query=sub_query,
                num_final_docs=1,
            )

            for doc in reranked:
                sample = doc["text"][:80]
                if sample not in seen_texts:
                    seen_texts.add(sample)
                    final_docs.append(doc)

        if not final_docs:
            return {"error": "No page content could be extracted"}

        # 7. RETURNING THE CONTEXT THAT CONTAINS THE ANSWERS
        return {"final_docs": final_docs}
    except Exception as e:
        return {"error": f"An error occured: {e}"}
