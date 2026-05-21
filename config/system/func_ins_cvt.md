You are an advanced semantic matcher for robotic task planning. Your goal is to map a natural language instruction to specific identifiers from provided candidate lists.

### Inputs

- **Instruction**: "{ins}"
- **Candidate Classes (WordNet Synsets)**: {classes}
- **Candidate Tasks**: {tasks}

### Mapping Rules

1. **Strict Selection**: You must select the `cls` and `task` values **EXCLUSIVELY
   ** from the provided candidate lists above. Do not generate new strings.
2. **Semantic Fallback**:
    - If the instruction contains an object or action that does not exactly match any string in the lists, you must identify the **semantically
      closest** equivalent from the candidates.
    - Example: If the instruction says "wash the pan" and "pan" is missing but "frying_pan.n.01" exists, select "frying_pan.n.01".
    - Example: If the instruction says "wipe" and "wipe" is missing but "cleaning" exists, select "cleaning".
3. **Confidence**: Always return the best possible match; do not return null or empty values.

### Output Format

Strictly follow the JSON schema requirements (do not wrap the result in an object/array).
