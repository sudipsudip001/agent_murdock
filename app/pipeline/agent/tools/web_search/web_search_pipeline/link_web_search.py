import asyncio
import logging
import os
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

EMAIL = os.getenv("EMAIL")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


class LinkWebSearch:
    def __init__(self) -> None:
        pass

    async def fetch_url(self, url: str) -> str:
        parsed = urlparse(url)
        if "wikipedia.org" in parsed.netloc and parsed.path.startswith("/wiki/"):
            title = parsed.path.replace("/wiki/", "")
            return await self.fetch_wikipedia_api(title)
        else:
            return await self.fetch_generic(url)

    async def fetch_generic(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MyApp/1.0)"}
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            # If we ended up at Wikipedia after a redirect, use the API instead
            final = str(resp.url)
            if "wikipedia.org" in final and "/wiki/" in final:
                title = urlparse(final).path.replace("/wiki/", "")
                return await self.fetch_wikipedia_api(title)
            return cast(str, resp.text)

    async def fetch_wikipedia_api(self, title: str) -> str:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        headers = {"User-Agent": f"MyApp/1.0 ({EMAIL})"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(api_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        return cast(str, page.get("extract", ""))

    async def search_urls(
        self,
        queries_string: list[str],
    ) -> list[dict[str, Any]]:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient() as client:
                search_tasks = [
                    client.post(SERPER_URL, json={"q": q}, headers=headers)
                    for q in queries_string
                ]
                search_responses = await asyncio.gather(*search_tasks)

                all_work_data = []
                for resp in search_responses:
                    resp.raise_for_status()
                    results = resp.json().get("organic", [])[:1]
                    for r in results:
                        all_work_data.append(
                            {
                                "title": r.get("title", ""),
                                "link": r.get("link", ""),
                                "body": r.get("snippet", ""),
                            }
                        )
                    logger.debug("Serper returned %d results", len(results))

            return all_work_data

        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code, detail="Serper API error"
            ) from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
