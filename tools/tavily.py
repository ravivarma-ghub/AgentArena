import os
from typing import Any, Dict, List, Optional
from config import TAVILY_API_KEY

class TavilyTool:
    """Tool wrapper around Tavily Search API with error handling and result formatting."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or TAVILY_API_KEY
        self._client = None
        if self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                print(f"[TavilyTool] Warning: Failed to initialize TavilyClient: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def search(self, query: str, max_results: int = 4, search_depth: str = "basic") -> Dict[str, Any]:
        """
        Execute web search and return structured results and formatted context.
        """
        if not self._client:
            return {
                "success": False,
                "error": "Tavily API key not configured or client unavailable.",
                "query": query,
                "results": [],
                "formatted_context": "[Tavily Search unavailable - proceeding with internal knowledge]"
            }

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth
            )
            raw_results = response.get("results", [])
            formatted_snippets = []
            structured_results = []

            for idx, item in enumerate(raw_results, 1):
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                content = item.get("content", "")
                structured_results.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
                formatted_snippets.append(
                    f"[{idx}] {title}\nURL: {url}\nExcerpt: {content}\n"
                )

            formatted_context = "\n".join(formatted_snippets) if formatted_snippets else "No search results found."

            return {
                "success": True,
                "query": query,
                "results": structured_results,
                "formatted_context": formatted_context
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "formatted_context": f"[Search error: {str(e)}]"
            }
