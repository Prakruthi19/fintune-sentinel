"""finbench: measure how well small LLMs solve FinQA questions by calling tools.

Stage 1 of FinTune-Sentinel. The model reads a filing excerpt, calls a
calculator / table tool for the arithmetic, and submits a final answer.
Nothing here trains a model; it produces the baseline numbers that decide
whether fine-tuning is worth doing at all.
"""
