# AI Use Disclosure — AI_A1_G15

This file discloses every generative AI tool used during the project,
as required by the SWE 3513 assignment rules.

## Tool 1: ChatGPT (OpenAI)

**Purpose:**
- Explanations of gradient descent derivation
- Help debugging pandas schema validation
- Suggestions for organizing the codebase

**Important prompts or requests:**
- "Explain batch gradient descent for linear regression from first principles"
- "How to validate a CSV schema in pandas and handle missing values"
- "How to compute silhouette score for K selection"

**Files affected:**
- `src/data.py` — schema validation and missing value handling
- `src/regression.py` — gradient descent derivation
- `src/clustering.py` — K selection logic

**How we verified the output:**
- Re-derived the MSE gradient by hand on paper
- Ran the pipeline on the issued dataset and inspected every output
- Cross-checked metrics with scikit-learn equivalents in a scratch script
- Reviewed every line before committing