# Notebook Review Workflow for Claude

## Standard Review Process

When reviewing new lecture or example notebooks, follow this workflow:

### Step 1: Read the New Notebook
- Read the unreleased lecture or example notebook provided by the user
- Get familiar with its content, structure, and current state

### Step 2: Review Recent Released Examples
- Read previously released example or lecture notebooks that match the topic/style
- Use these as reference for tone, style, and formatting

### Step 3: Generate Learning Objectives and Summary Section
Generate three components:

#### A. Learning Objectives (3 items)
- Create three specific, actionable learning objectives
- Use format: `* __[Objective title]:__ [description]`
- Each should be concrete and measurable
- **Important**: Check the notebook for specific instructions or placeholders in the learning objectives section - the user may specify exactly what should go in each objective
- Only use direct, simple, and concise language
- Avoid adjectives
- Content must be directly supported by and derived from the notebook content - do not generate false or unsupported text

#### B. Summary Section with Key Takeaways
- Add a **Summary** section after the main content
- Include **Key Takeaways** subsection with exactly three bullet points
- Use format:
  ```
  * **[Takeaway title]:** [explanation]
  ```
- Each takeaway should be concise (1-2 sentences)
- **Important**: Check the notebook for specific instructions or placeholders in the summary section - the user may specify exactly what should go in each takeaway
- Only use direct, simple, and concise language
- Avoid adjectives
- Content must be directly supported by and derived from the notebook content - do not generate false or unsupported text

#### C. Structure
- Add a **Direct Summary Sentence**: One sentence that captures the core essence of the notebook
- Follow with **Conclusion Sentence**: One sentence about implications or next steps
- These should appear before or after the Key Takeaways section

**Tone & Style Requirements:**
- Match the tone, style, and formatting from the uploaded reference examples
- Use the same technical language and depth as reference materials
- Maintain consistent markdown formatting
- Ensure mathematical notation and code examples follow established patterns
- Use direct, simple, and concise language
- Avoid unnecessary adjectives

### Step 4: Mathematical Notation Review
Check all mathematical notation for consistency and precision:

**Notation Consistency:**
- Verify all norms are explicitly specified (e.g., $\|\cdot\|_{2}$ for 2-norm, $\|\cdot\|_{2}^{2}$ for squared 2-norm, $\|\cdot\|_{1}$ for 1-norm)
- Do not use unspecified norms like $\|\cdot\|$ without a subscript
- Ensure vector/matrix notation is consistent throughout (e.g., always use $\mathbf{x}$ for vectors, $\mathbf{W}$ for matrices)
- Check that transpose notation is consistent (choose one style and use throughout)
- Verify superscript/subscript conventions are applied uniformly

**Convergence and Stopping Criteria:**
- All convergence criteria must specify the exact norm or metric used
- Define tolerance parameters explicitly (e.g., $\epsilon > 0$)
- Ensure all variables in convergence checks are defined before use
- Specify whether comparing to previous iteration, initial value, or fixed point

**Algorithm Specifications:**
- All algorithm steps must be unambiguous and executable
- Define all variables before first use
- Specify data types where relevant (e.g., $\theta\in\mathbb{R}^{p}$)
- Include initialization requirements
- Clarify loop bounds and termination conditions

**Domain and Range:**
- Specify function domains and codomains (e.g., $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$)
- Ensure activation function signatures match their descriptions
- Verify set membership notation is correct

### Step 5: Final Quality Check
- Fix all spelling errors
- Fix all grammar issues
- Fix punctuation errors
- Fix awkward or unclear text
- Ensure consistency with reference examples
- Verify technical accuracy
- **Language quality**: Verify content uses direct, simple, and concise language without unnecessary adjectives
- **Content accuracy**: Confirm all generated content is directly supported by the notebook - no unsupported claims
- **Mathematical rigor**: Confirm all notation issues from Step 4 have been addressed

### Step 6: Rate the Notebook
Provide a rating on a scale of **0-10** based on:

**Quality Criteria:**
- **Flow**: Does the notebook have a clear logical progression?
- **Technical Correctness**: Are all technical concepts accurate?
- **Mathematical Correctness**: Are all equations and mathematical statements correct?
- **Mathematical Precision**: Are norms, convergence criteria, and technical specifications explicit and unambiguous?
- **Clarity**: Is the content understandable to the target audience?
- **Completeness**: Are all necessary components included?

**Rating Scale:**
- 0-2: Really bad (major issues throughout)
- 3-4: Poor (significant problems)
- 5-6: Adequate (acceptable but needs improvement)
- 7-8: Good (mostly correct with minor issues)
- 9-10: Really great (excellent, minimal or no issues)

**Provide:**
- Overall rating (e.g., `8/10`)
- Brief explanation of rating
- 2-3 specific strengths
- 1-2 specific areas for improvement (if any)

---

## Critical Guidelines

### Language Standards
- Use direct, simple, and concise language at all times
- Avoid adjectives unless essential for technical accuracy
- Remove any flowery, descriptive, or vague language
- Prefer active voice and clear subject-verb-object structure

### Content Integrity
- All generated content (learning objectives, summaries, takeaways) must be directly supported by the notebook
- Do not generate unsupported text, claims, or ideas
- Do not extrapolate beyond what is explicitly covered in the notebook
- If unsure whether content is supported, do not include it
- Flag any areas where you cannot find supporting evidence in the notebook

### Handling Placeholders and Instructions
- Check all sections of the notebook carefully for existing instructions or placeholders
- Users may leave specific guidance about what should go in learning objectives or summary sections
- Follow these instructions precisely - they override general formatting guidelines
- Ask for clarification if instructions are ambiguous
