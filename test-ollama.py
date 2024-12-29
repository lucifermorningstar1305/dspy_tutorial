import dspy

if __name__ == "__main__":
    lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    print(lm("Say 'this is a test!'", temperature=0.7))

    qa = dspy.ChainOfThought("question -> answer")
    response = qa(question="Can anyone beat Meta in social networking platform?")
    print(response.answer)
    print(lm.history)
