import hashlib
from urllib.parse import urlparse, urlunparse

from url_normalize import url_normalize


class LinkDeduplicator:
    def __init__(self) -> None:
        pass

    def _strip_trailing_slash(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.path.endswith("/") and parsed.path != "/":
            stripped = parsed._replace(path=parsed.path.rstrip("/"))
            return urlunparse(stripped)
        return url

    def get_canonical_url(self, urls: list[str]) -> dict[str, str]:
        mapping = {}
        for url in urls:
            norm = url_normalize(url)
            norm = self._strip_trailing_slash(norm)
            mapping[url] = norm
        return mapping

    def deduplicate(self, urls: list[str]) -> set[str]:
        original_to_norm = self.get_canonical_url(urls)
        seen_hashes = {}

        for original, norm in original_to_norm.items():
            url_hash = hashlib.md5(norm.encode(), usedforsecurity=False).hexdigest()
            if url_hash not in seen_hashes:
                seen_hashes[url_hash] = original

        return set(seen_hashes.values())
