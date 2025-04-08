import logging
import time
from functools import wraps
from typing import Callable, Any, Dict

# 로거 설정
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """
    로깅 설정
    
    Args:
        level: 로깅 레벨
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"유효하지 않은 로깅 레벨: {level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def timer(func: Callable) -> Callable:
    """
    함수 실행 시간을 측정하는 데코레이터
    
    Args:
        func: 시간을 측정할 함수
        
    Returns:
        Callable: 래핑된 함수
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"함수 '{func.__name__}' 실행 시간: {elapsed_time:.2f}초")
        return result
    return wrapper

def format_response(state_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    API 응답 형식으로 상태 데이터 포맷팅
    
    Args:
        state_data: 포맷팅할 상태 데이터
        
    Returns:
        Dict[str, Any]: 포맷팅된 응답 데이터
    """
    # 필요한 필드만 추출
    response = {
        "answer": state_data.get("answer", ""),
        "sources": state_data.get("sources", [])
    }
    
    # 오류가 있는 경우
    if error := state_data.get("error"):
        response["success"] = False
        response["error"] = error
    else:
        response["success"] = True
    
    return response

def create_error_response(message: str) -> Dict[str, Any]:
    """
    오류 응답 생성
    
    Args:
        message: 오류 메시지
        
    Returns:
        Dict[str, Any]: 오류 응답
    """
    return {
        "success": False,
        "error": message,
        "answer": "",
        "sources": []
    }