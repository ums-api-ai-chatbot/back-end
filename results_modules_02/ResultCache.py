import hashlib
from typing import Dict


class ResultCache:
    """질문-응답 결과를 캐싱하는 클래스"""
    
    def __init__(self):
        self.cache: Dict[str, str] = {}
        
    def _hash_key(self, question: str) -> str:
        """질문을 해시하여 고유 키 생성"""
        return hashlib.sha256(question.encode()).hexdigest()
    
    def get(self, question: str) -> str:
        """캐시에서 결과 가져오기"""
        key = self._hash_key(question)
        return self.cache.get(key)
    
    def set(self, question: str, response: str):
        """캐시에 결과 저장"""
        key = self._hash_key(question)
        self.cache[key] = response
    
    def clear(self):
        """캐시 초기화"""
        self.cache.clear()