
import modal

f = modal.Function.lookup("example-vllm-mixtral", "Model.completion_stream")

for text in f.remote_gen("What is the story about the fox and grapes?"):
	print(text, end="", flush=text.endswith("\n"))

"""
Results after running:

 The story of the "Fox and Grapes" is a fable attributed to Aesop, a slave and storyteller believed to have lived in ancient Greece around 600 BCE. The fable tells the tale of a fox who sees some high-hanging grapes and wishes to eat them. However, the fox is unable to reach the grapes, despite making several attempts.

Feeling defeated, the fox walks away with his head held low. Then, he turns around and says, "The grapes are sour, and not worth having!"

This fable
	Generated 128 tokens from mistralai/Mixtral-8x7B-Instruct-v0.1 in 4.3s, throughput = 30 tokens/second on GPU(A100-80GB, count=2).

"""