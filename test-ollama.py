import dspy

if __name__ == "__main__":
    lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    print(lm("Say 'this is a test!'", temperature=0.7))

    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question="What is an Imperative computation grpah?")
    print(response.answer)
    # print(lm.history)
