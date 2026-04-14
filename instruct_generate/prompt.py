question_prompt1 = """
**Instruction**:
Generate a reasonable question based on the entities and source and the following rules. Only generate questions, no answers:
**Rules**:

Based on the provided text and the examples I've given, generate questions that align with the following task types:

**Examples**: 
1. **Yes-or-No Question Answering**: 
   1."The recipient can use the information for other purposes"? Yes, no, or maybe? 

2. **Multiple-Choice Question Answering**: 
   1.The recipient can use the information for whatever he wants. Is this always, sometimes, or never correct? 

3. **Extractive Question Answering**: 
   1.Which sentence clearly restricts the recipient's use of the confidential information? 

4. **Natural Language Inference**:
   1. ...
   2. ...

5. **Sentiment Analysis**:
   ...

6. **Topic Classification**:
   ...
For each task type, generate **3 questions** following the same requirements. And strictly follow the format of {X. **type of question**}

Please ensure the questions are based on the provided text and adhere to the characteristics of each task type.
"""

question_prompt2 = """
**Instruction**:
Generate three questions from <Source Sentences> based on the provided Rules and the given Examples. Your generated question should reflect the logic of the Source text and conform to the Yes‑or‑No Question Answering task type.
**Rules**:
Task Type: Yes-or-No Question Answering
The question must be logically inferable from the input Source sentence(s).
The expected answer format conclude but not limited to:
Yes, no, or maybe?
True, False, or Neither?
When multiple sentences are provided as Source, consider them together to form a meaningful inference-based question.
The phrasing of the question should be clear, formal, and grammatically correct.

*Examples*: 
**Source Sentences**:
“The occurrence of toxicity displays a marked interindividual variation, and for this reason the pharmacokinetics and pharmacodynamics of anthracyclines have been extensively investigated in order to identify integrated models that can be used in the clinical setting to prevent the development of serious toxicity, mainly leucopenia, and maximise tumour exposure. Pharmacokinetics has been recognised to influence both the toxicity and the activity of anthracyclines; in particular, there is increasing evidence that the mode of administration plays an important role for cumulative cardiotoxicity and data indicate that bolus administration, rather than continuous infusion, appears to be an important risk factor for anthracycline-induced cardiomyopathy, thus implying that this type of toxicity is maximum concentration-dependent. On the contrary, exposure to the drug, as measured by area under the curve, seems best related to the occurrence of leucopenia. Finally, the development of pharmacokinetic-pharmacodynamic models allows the simulation of drug effects and ultimately dose optimisation in order to anticipate important toxicities and prevent their occurrence by the administration of prophylactic treatments.” ,
“Therefore, a physiologically based pharmacokinetic (PBPK) model was developed to assess the potential for drug-drug interactions (DDIs) between orteronel and theophylline, repaglinide, (S)-warfarin and omeprazole, which are sensitive substrates of CYP1A2, 2C8, 2C9 and 2C19, respectively. Simulation of the area under the plasma concentration-time curve (AUC) of these four CYP substrates in the presence and absence of orteronel revealed geometric mean AUC ratios <1.25. Therefore, in accordance with the 2012 US FDA Draft Guidance on DDIs, orteronel can be labeled a 'non-inhibitor' and further clinical DDI evaluation is not required. In PBPK models of moderate and severe renal impairment, the AUC of orteronel was predicted to increase by 52% and 83%, respectively. These results are in agreement with those of a clinical trial in which AUC increases of 38% and 87% were observed in patients with moderate and severe renal impairment, respectively.”
**Question**:

1. **Single Sentence-based on sentence1 in Source**: 
   Does the pharmacokinetics of anthracyclines influence both their toxicity and activity? True, False, or Neither?

2. **Single Sentence-based on sentence2 in Source**: 
   Can orteronel be classified as a 'non-inhibitor' of CYP1A2, 2C8, 2C9, and 2C19 based on PBPK simulations? Yes, no, or maybe?

3. **Composite Two‑sentence Integration in Source**: 
   Is it reasonable to conclude that “while anthracycline toxicity is influenced by mode of administration and exposure, orteronel’s interaction potential is sufficiently characterized by PBPK modeling to waive further clinical DDI studies”? Yes, no, or maybe?

Please strictly follow the format of {X. **type of question**} and ensure the questions are based on the provided text and adhere to the characteristics of each task type.
"""

question_prompt3 = """

**Persona**
You are an expert Legal Content Analyst. Your primary skill is to meticulously analyze complex legal texts and extract precise, logically inferable questions that test comprehension and highlight key clauses. You are detail-oriented, formal, and systematic in your approach.

**Objective**
Generate three distinct Yes/No questions based on the provided `<Source Sentences>`, adhering strictly to the specified rules, types, and output format.

**Instructions**
1.  **Analyze the `<Source Sentences>`**: Carefully read and understand the provided legal text.
2.  **Generate Question 1 (Single Sentence - Direct Inference)**:
    *   **Reasoning**: First, write a brief, one-sentence explanation of the direct inference you are drawing from the *first* source sentence.
    *   **Question**: Then, formulate a question based on this inference. This question should be a straightforward verification of a statement made in that sentence.
3.  **Generate Question 2 (Single Sentence - Implication)**:
    *   **Reasoning**: First, write a brief, one-sentence explanation of the logical implication you are deriving from the *second* source sentence.
    *   **Question**: Then, formulate a question that tests this implication.
4.  **Generate Question 3 (Composite - Synthesis)**:
    *   **Reasoning**: First, write a brief, one-sentence explanation of how you are synthesizing information from *both* source sentences to create a new, combined logical point.
    *   **Question**: Then, formulate a question based on this synthesized logic.
5.  **Adhere to Format**: Ensure your final output strictly follows the structure and numbering demonstrated in the examples below.

**Rules & Constraints**
*   **Task Type**: All questions must be answerable with "Yes," "No," or "Maybe." The required answer format, appended to the question, is: `Yes, no, or maybe?`
*   **Logical Inference**: Questions must be strictly and logically inferable from the provided source text alone. Do not introduce or rely on any external knowledge.
*   **Clarity and Formality**: The phrasing of each question must be clear, formal, and grammatically impeccable.
*   **Avoid Triviality**: Do not simply rephrase a sentence as a question. The question should require a degree of logical deduction.

---
**Examples**

**Source Sentences**:
“Confidential Information does not include information that the Recipient demonstrates (a) is in the public domain through no fault of, or disclosure by, the Recipient or its Representatives, subsidiaries or affiliates, (b) was properly known to the Recipient, without restriction, prior to disclosure by the Disclosing Party, (c) was properly disclosed to the Recipient by another person, but only if such person is not bound by a confidentiality agreement with the Disclosing Party or is not otherwise restricted from providing such information by a contractual, legal or fiduciary duty. ...” ,
“If either Party decides not to proceed with the Opportunity, the Parties will promptly return or destroy all Confidential Information received under this Agreement, and all copies, extracts and other objects or items in which such Confidential Information may be contained or embodied, and certify in writing that it has complied with this requirement.”

**Generated Output**:

1.  **Single Sentence-based on sentence 1 in Source**:
    *   **Reasoning**: The first sentence explicitly excludes information already in the public domain from the definition of "Confidential Information," provided the recipient is not at fault.
    *   **Question**: Based on the context above, should we assume that "information publicly available through no action of the recipient is not considered Confidential Information"? Yes, no, or maybe?

2.  **Single Sentence-based on sentence 2 in Source**:
    *   **Reasoning**: The second sentence creates a direct obligation for a party who withdraws to both destroy confidential information and provide written certification of that destruction.
    *   **Question**: According to the statement above, when a Party decides not to proceed, is it sufficient for them to destroy all Confidential Information without providing written certification? Yes, no, or maybe?

3.  **Composite Two‑sentence Integration in Source**:
    *   **Reasoning**: The first sentence defines what is *not* confidential, while the second sentence mandates the destruction of all *received* "Confidential Information." Synthesizing these, the obligation at the point of withdrawal applies to everything received *as* confidential, regardless of its ultimate status.
    *   **Question**: Based on the combined context, if a Recipient receives information they believe was already in the public domain, are they still obligated to return or destroy it upon withdrawal if it was provided to them as "Confidential Information" under the agreement? Yes, no, or maybe?

---
Now, use the framework above to generate three questions from the following source sentences.

**<Source Sentences>**:


"""



relevant_prompt = """
# Role: Expert Content Analyst

You are an expert content analyst specializing in thematic analysis. Your analysis must be objective and strictly based on the provided texts and the scoring rubric below. Do not use external knowledge or make inferences beyond the explicit content of the texts.

---
## Core Task

Your task is to determine if **Text A** and **Text B** discuss the **same primary subject matter or core concept**. You must answer with a definitive "Yes" or "No".

---
## Scoring Rubric

1.  **Answer "Yes" if:**
    - Both texts directly discuss, define, or describe different aspects of the *same specific, narrowly-defined concept*. There must be a clear and direct thematic overlap on the primary subject.

2.  **Answer "No" if:**
    - The texts discuss different concepts, even if they fall under the same broad category (e.g., both are about data privacy, but one concerns disclosure policies and the other payment collection).
    - The thematic link is indirect or requires external knowledge to connect.

---
## Examples

**Example 1:**

**Text A:** “Confidential Information does not include information that the Recipient demonstrates (a) is in the public domain through no fault of, or disclosure by, the Recipient or its Representatives, subsidiaries or affiliates, (b) was properly known to the Recipient, without restriction, prior to disclosure by the Disclosing Party, (c) was properly disclosed to the Recipient by another person, but only if such person is not bound by a confidentiality agreement with the Disclosing Party or is not otherwise restricted from providing such information by a contractual, legal or fiduciary duty. …”
**Text B:** “If either Party decides not to proceed with the Opportunity, the Parties will promptly return or destroy all Confidential Information received under this Agreement, and all copies, extracts and other objects or items in which such Confidential Information may be contained or embodied, and certify in writing that it has complied with this requirement.”
**Rationale:** Both texts are explicitly about the definition, exceptions, and handling of "Confidential Information." The core concept is identical.
**Answer:** Yes

**Example 2:**

**Text A:** “The information may be disclosed to: (i) provide joint content and our services (eg, registration, coordination of membership accounts between the Viber corporate family, transactions, analytics and customer support); (ii) help detect and prevent potentially illegal acts, violations of our policies, fraud and/or data security breaches.”
**Text B:** “When you order any good or service through the Game, including any virtual currency or virtual good, our payment processing service provider will collect your name, phone number, e-mail address, mailing address, billing address, and complete credit card information that enables them to receive your payment.”
**Rationale:** While both texts relate to user data, Text A discusses the *disclosure* of data for service provision, whereas Text B discusses the *collection* of specific data for payment processing. These are two different concepts within the broader topic of data handling.
**Answer:** No

---
## Your Turn

**Text A:** {text_a}
**Text B:** {text_b}
**Answer:**
"""
relevant_prompt1 = """
# Role: Expert Content Analyst

You are an expert content analyst specializing in thematic analysis and multi-hop reasoning assessment. Your analysis must be objective and based on the provided texts.

---
## Core Task

Determine if **Text A** and **Text B** are related enough to be used for multi-hop task generation. You must answer with a definitive "Yes" or "No".

---
## Scoring Rubric (Very Relaxed Criteria)

**Answer "Yes" if ANY of the following apply:**

- Both texts contain any shared words, concepts, or themes (even loosely related)
- The texts could potentially come from the same source or document type
- There is any possible connection that could be made between the texts
- The texts discuss topics within the same broad field (e.g., both about technology, law, privacy, business, etc.)
- The texts could be used together to create any type of reasoning question
- There is any overlap in subject matter, no matter how minor
- The texts share any common entities, terms, or ideas

**Answer "No" ONLY if:**
- The texts are absolutely, completely unrelated with zero possible connections
- The texts come from entirely different universes of discourse with no imaginable link
- It would be impossible to create any meaningful question using both texts

**Default to "Yes" when uncertain.**

---
## Examples

**Example 1:**

**Text A:** 'The remainder of this Privacy Policy sets out further information regarding some of these categories.'
**Text B:** 'In addition, this information helps us track any fraudulent activities and other inappropriate activities and monitor content integrity; or'
**Answer:** Yes

**Example 2:**

**Text A:** 'Lima Sky, LLC (Lima Sky) has adopted this privacy policy (Privacy Policy) to explain how Lima Sky collects, stores, and uses the information collected in connection with Lima Skys products, services, and websites (together Services).'
**Text B:** 'Thereafter if the collected data is no longer needed for purposes specified in this Privacy Policy, Lima Sky deletes all aforementioned data in its possession within a reasonable timeframe.', 'The remainder of this Privacy Policy sets out further information regarding some of these categories.'
**Answer:** Yes

---
## Your Turn

**Text A:** {text_a}
**Text B:** {text_b}
**Answer:**
"""
