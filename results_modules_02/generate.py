class Generate:
    """답변 생성 기능을 담당하는 클래스"""
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def execute(self, state):
        print("==== [GENERATE] ====")
        question = state["question"]
        documents = state["documents"]

        # RAG 답변 생성
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"generation": generation}