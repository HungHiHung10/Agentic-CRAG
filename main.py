from modules.rag.service import build_crag_workflow


def main():
    graph = build_crag_workflow()
    query = "Tri\u1ec7u ch\u1ee9ng v\u00e0 c\u00e1ch \u0111i\u1ec1u tr\u1ecb b\u1ec7nh Tr\u00fang th\u1eed?"
    print(f"\nQuestion: {query}")
    result = graph.invoke({"question": query, "documents": [], "generation": ""})
    print(f"\nAnswer:\n{result['generation']}")


if __name__ == "__main__":
    main()