import time
from typing import Dict, Any, Tuple, Optional, List
import logging
from collections import OrderedDict
import hashlib
import json

logger = logging.getLogger(__name__)

class ChatResponseCache:
    """
    챗봇 응답을 캐싱하는 클래스.
    LRU(Least Recently Used) 정책으로 캐시 크기를 관리하고,
    원본 질문과 재작성된 질문 모두를 고려합니다.
    """
    def __init__(self, max_size: int = 1000, ttl: int = 86400):
        """
        Args:
            max_size: 캐시에 저장할 최대 항목 수
            ttl: 캐시 항목의 유효 시간(초), 기본값은 24시간
        """
        self.cache = OrderedDict()  # 순서가 있는 딕셔너리로 LRU 구현
        self.query_mapping = {}  # 재작성된 질문과 원본 질문의 매핑
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        질문과 채팅 이력을 기반으로 고유한 캐시 키를 생성합니다.
        
        Args:
            query: 사용자 질문
            chat_history: 채팅 이력 (선택 사항)
            
        Returns:
            생성된 캐시 키
        """
        # 채팅 이력이 없는 경우 빈 리스트로 초기화
        if chat_history is None:
            chat_history = []
        
        # 질문 정규화 (소문자, 공백 제거)
        normalized_query = query.strip().lower()
        
        # 질문과 채팅 이력을 포함한 딕셔너리 생성
        cache_dict = {
            "query": normalized_query,
            "chat_history": chat_history
        }
        
        # 딕셔너리를 JSON 문자열로 변환하고 해시 생성
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """
        캐시에서 응답을 검색합니다.
        
        Args:
            query: 사용자 질문
            chat_history: 채팅 이력 (선택 사항)
            
        Returns:
            캐시된 응답 또는 None (캐시 미스)
        """
        key = self._generate_key(query, chat_history)
        
        if key in self.cache:
            # 캐시 항목 가져오기
            timestamp, response = self.cache[key]
            
            # TTL 확인
            if time.time() - timestamp <= self.ttl:
                # 캐시 히트 - LRU 갱신을 위해 항목을 제거하고 다시 추가
                self.cache.pop(key)
                self.cache[key] = (timestamp, response)
                self.hits += 1
                logger.info(f"Cache hit for query: {query[:30]}...")
                return response
            else:
                # 캐시 항목이 만료됨
                self.cache.pop(key)
                if key in self.query_mapping:
                    del self.query_mapping[key]
                self.misses += 1
                logger.info(f"Cache expired for query: {query[:30]}...")
                return None
                
        # 재작성된 질문을 통해 원본 질문의 캐시 키를 찾아봄
        normalized_query = query.strip().lower()
        for original_key, rewritten_queries in self.query_mapping.items():
            if normalized_query in rewritten_queries:
                if original_key in self.cache:
                    timestamp, response = self.cache[original_key]
                    # TTL 확인
                    if time.time() - timestamp <= self.ttl:
                        # 캐시 항목 재활용
                        self.hits += 1
                        logger.info(f"Cache hit via rewritten query mapping: {query[:30]}...")
                        return response
        
        self.misses += 1
        logger.info(f"Cache miss for query: {query[:30]}...")
        return None
    
    def set(self, query: str, response: Dict[str, Any], 
            chat_history: Optional[List[Dict[str, Any]]] = None,
            rewritten_query: Optional[str] = None) -> None:
        """
        응답을 캐시에 저장합니다.
        
        Args:
            query: 사용자 원본 질문
            response: 저장할 응답
            chat_history: 채팅 이력 (선택 사항)
            rewritten_query: 재작성된 질문 (있는 경우)
        """
        key = self._generate_key(query, chat_history)
        
        # 캐시가 최대 크기에 도달한 경우 가장 오래된 항목 제거
        if len(self.cache) >= self.max_size:
            oldest_key, _ = self.cache.popitem(last=False)  # 가장 오래된 항목(첫 번째) 제거
            if oldest_key in self.query_mapping:
                del self.query_mapping[oldest_key]
        
        # 현재 시간과 함께 응답 저장
        self.cache[key] = (time.time(), response)
        
        # 재작성된 질문이 있는 경우 매핑 저장
        if rewritten_query and rewritten_query.strip().lower() != query.strip().lower():
            normalized_rewritten = rewritten_query.strip().lower()
            if key not in self.query_mapping:
                self.query_mapping[key] = set()
            self.query_mapping[key].add(normalized_rewritten)
            logger.info(f"Mapped rewritten query: {rewritten_query[:30]}... -> {query[:30]}...")
        
        logger.info(f"Cached response for query: {query[:30]}...")
    
    def clear(self) -> None:
        """캐시를 비웁니다."""
        self.cache.clear()
        self.query_mapping.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계를 반환합니다."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "rewrite_mappings": len(self.query_mapping),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl": self.ttl
        }