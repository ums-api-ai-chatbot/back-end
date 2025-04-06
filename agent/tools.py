import logging
from typing import List, Dict, Any, Optional
import json

from langchain_community.tools import DuckDuckGoSearchRun
from tavily import TavilyClient
from config import TAVILY_API_KEY, SEARCH_TOOL, SEARCH_TOP_K

# 로깅 설정
logger = logging.getLogger(__name__)

class InternetSearchTool:
    """인터넷 검색을 위한 툴 클래스"""
    
    def __init__(self):
        self.search_tool = SEARCH_TOOL
        logger.info(f"Initializing Internet Search Tool using {self.search_tool}")
        
        if self.search_tool == "tavily":
            if not TAVILY_API_KEY:
                logger.warning("Tavily API key not found, defaulting to DuckDuckGo")
                self.search_tool = "duckduckgo"
            else:
                self.client = TavilyClient(api_key=TAVILY_API_KEY)
                
        if self.search_tool == "duckduckgo":
            self.client = DuckDuckGoSearchRun()
    
    def search(self, query: str, num_results: int = SEARCH_TOP_K) -> List[Dict[str, Any]]:
        """
        인터넷 검색을 실행하고 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리
            num_results: 반환할 검색 결과 수
            
        Returns:
            검색 결과 목록
        """
        logger.info("Agent : Internet Searching : " + query)

        try:
            if self.search_tool == "tavily":
                return self._tavily_search(query, num_results)
            else:
                return self._duckduckgo_search(query, num_results)
        except Exception as e:
            logger.error(f"Error in internet search: {e}")
            return []
    
    def _tavily_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Tavily API를 사용한 검색"""
        logger.info(f"Performing Tavily search for: {query}")
        
        try:
            # Tavily 검색 실행
            search_result = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results
            )
            
            # 결과 형식화
            results = []
            for item in search_result.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "score": item.get("score", 0)
                })
                
            logger.info(f"Found {len(results)} results from Tavily")
            return results
            
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []
    
    def _duckduckgo_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색 실행"""
        logger.info(f"Performing DuckDuckGo search for: {query}")
        
        try:
            # DuckDuckGo 검색 실행
            search_results = self.client.run(query)
            
            # 결과 파싱 시도
            try:
                results = json.loads(search_results)
            except json.JSONDecodeError:
                results = self._parse_duckduckgo_results(search_results, num_results)
                
            logger.info(f"Found {len(results)} results from DuckDuckGo")
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _parse_duckduckgo_results(self, results_text: str, num_results: int) -> List[Dict[str, Any]]:
        """DuckDuckGo 검색 결과 텍스트를 파싱합니다."""
        # 결과가 문자열로 반환될 경우 간단하게 파싱
        lines = results_text.split("\n")
        results = []
        
        for line in lines[:num_results]:
            results.append({
                "title": "",  # DuckDuckGo는 별도의 타이틀을 제공하지 않음
                "content": line,
                "url": "",  # URL 정보가 없을 수 있음
                "score": 0.5  # 기본 점수
            })
            
        return results