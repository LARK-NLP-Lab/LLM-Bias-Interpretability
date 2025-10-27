# LLM-Bias-Interpretability

## Investigating Racial Bias in Large Language Models (LLMs)

We employ linear probing on a Qwen model, using race-conditioned medical prompts to analyze layer-specific MLP neuron activations. Our goal is to pinpoint "bias-related neurons" through differential activation and neuron activation calculations. We then use a sign-aware Logit Lens to semantically decode these neurons' contributions, aiming to understand bias propagation and inform mitigation strategies.


`modeling_qwen2_mlp.py`: Custom Qwen model for MLP output.
