question_prompt1 = """
**Instruction**:
Generate three questions from the <Source Sentences> based on the provided Rules and Examples. Each question must be a closed-ended, logical inquiry that can be definitively answered based only on the provided text.
**Rules**:
Rules:

1. Task Type: Three-Option Logical Inquiry.
2. Logical Inference: The question must be logically inferable from the input Source sentence(s). Do not use outside knowledge.
3. Question Suffix: The question must end with a three-option logical suffix. Choose one of the following formats:
Yes, no, or maybe?
True, False, or Neither?
Correct, Incorrect, or Uncertain?
4. Interrogative Structure: Use different interrogative structures (e.g., “Does…?”, “Is…?”, “Was…?”, “Have…?”, etc.) or frame the question around a statement.
5. Clarity and Grammar: The phrasing of the question should be clear, formal, and grammatically correct.

**Examples Start**: 
***Source Sentences***:
“Rationale and design of LAPLACE-2: a phase 3, randomized, double-blind, placebo- and ezetimibe-controlled trial evaluating the efficacy and safety of evolocumab in subjects with hypercholesterolemia on background statin therapy. Low-density lipoprotein cholesterol (LDL-C) levels are significantly associated with atherosclerotic cardiovascular disease (ASCVD) risk, and studies using interventions that lower LDL-C levels have been shown to reduce the risk of ASCVD events and mortality. Statin treatment is the current first-line therapy for lowering LDL-C and reducing ASCVD risk. However, many patients are still unable to reach recommended LDL-C goals on maximally tolerated statin therapy.”

***Question 1***:
Is statin treatment currently considered the first-line therapy for lowering LDL-C levels and reducing ASCVD risk? Yes, no, or maybe?
***Answer 1***: Yes.

***Question 2***:
Does maximally tolerated statin therapy always enable patients to achieve their recommended LDL-C goals? Yes, no, or maybe?
***Answer 2***:No.

***Question 3***:
The LAPLACE-2 trial was a phase 1 study evaluating the efficacy of evolocumab. True, False, or Neither?
***Answer 3***:False.

**Examples End**
Please strictly follow the format of {***Question n***:} and ensure the questions are based on the provided text and adhere to the characteristics of each task type.
"""

question_prompt2 = """
**Instruction**:
The <Source Texts> block below contains two distinct and unrelated passages. Your task is to generate three questions that require synthesizing, comparing, or relating information from both passages to be answered. Each question must be a closed-ended, logical inquiry that can be definitively answered based only on the provided text.
**Rules**:
Rules:
1. Synthesis Requirement: Each question must require the reader to process and integrate information from both distinct texts provided. Questions that can be answered using only one of the texts are not acceptable.
2. Task Type: Three-Option Logical Inquiry.
3. Logical Inference: The question must be logically inferable from the input Source sentence(s). Do not use outside knowledge.
4. Question Suffix: The question must end with a three-option logical suffix. Choose one of the following formats:
Yes, no, or maybe?
True, False, or Neither?
Correct, Incorrect, or Uncertain?
5. Interrogative Structure: Use different interrogative structures (e.g., “Does…?”, “Is…?”, “Was…?”, “Have…?”, etc.) or frame the question around a statement.
6. Clarity and Grammar: The phrasing of the question should be clear, formal, and grammatically correct.

**Examples Start**: 
***Source Sentences***:
<Text 1>
“Cornea transplantation was the recommendation by multiple cornea specialists as the treatment of choice. We decided prior to considering a transplant to employ the Athens Protocol (combined topography-guided partial PRK and CXL) in the right eye in February 2010 and in the left eye in September 2010. The treatment plan for both eyes was designed on the topography-guided wavelight excimer laser platform. RESULTS: Fifteen months after the right eye treatment, the right cornea had improved translucency and was topographically stable with uncorrected distance visual acuity (UDVA) 20/50 and CDVA 20/40 with refraction +0.50, -2.00 at 5°. We noted a similar outcome after similar treatment applied in the left eye with UDVA 20/50 and CDVA 20/40 with -0.50, -2.00 at 170° at the 8-month follow-up. CONCLUSION: In this case, the introduction of successful management of severe cornea abnormalities and scarring with the Athens Protocol may provide an effective alternative to other existing surgical or medical options.”
<Text 2>
“FISH studies were performed in 7 cases using the LSI BCR/ABL ES probe allowing the detection of the fusion BCR/ABL gene on the Ph chromosome in all of them and 9q34 deletions in 2 cases. Three cryptic complex rearrangements were detected by FISH studies. The third and the fourth chromosome regions involved in the 8 complex variant translocations were: 1q21, 1p36, 5q31, 11q13, 12q13, 12p13, and 20q12. In conclusion, FISH studies have been useful in the detection of the BCR/ABL rearrangements and 9q34 deletions, and to identify complex rearrangements that differ from the ones previously established by conventional cytogenetics.”

***Question 1***:
The provided text suggests that FISH studies were used as part of the Athens Protocol to manage severe cornea abnormalities. True, False, or Neither?
***Answer 1***: False.

***Question 2***:
Does the text confirm that the patient who received treatment for cornea abnormalities in 2010 was one of the 7 cases who underwent FISH studies for BCR/ABL rearrangements? Yes, no, or maybe?
***Answer 2***:Maybe.

***Question 3***:
Is it correct that the source text as a whole describes both a specific ophthalmological treatment protocol and a genetic analysis technique, with each description ending in a formal conclusion? Correct, Incorrect, or Uncertain?
***Answer 3***:Correct.

**Examples End**
Please strictly follow the format of {***Question n***:} and ensure the questions must synthesize information from both source texts to be answered, adhering to all rules.
"""

question_prompt3 = """
**Instruction**:
The <Source Texts> block below contains three distinct and unrelated passages. Your task is to generate three questions that require synthesizing, comparing, or relating information from all three passages to be answered. Each question must be a closed-ended, logical inquiry that can be definitively answered based only on the provided text.
**Rules**:
Rules:
1. Synthesis Requirement: Each question must require the reader to process and integrate information from all three distinct texts provided. Questions that can be answered using only one of the texts are not acceptable.
2. Task Type: Three-Option Logical Inquiry.
3. Logical Inference: The question must be logically inferable from the input Source sentence(s). Do not use outside knowledge.
4. Question Suffix: The question must end with a three-option logical suffix. Choose one of the following formats:
Yes, no, or maybe?
True, False, or Neither?
Correct, Incorrect, or Uncertain?
5. Interrogative Structure: Use different interrogative structures (e.g., “Does…?”, “Is…?”, “Was…?”, “Have…?”, etc.) or frame the question around a statement.
6. Clarity and Grammar: The phrasing of the question should be clear, formal, and grammatically correct.

**Examples Start**: 
***Source Sentences***:
<Text 1>
“Cornea transplantation was the recommendation by multiple cornea specialists as the treatment of choice. We decided prior to considering a transplant to employ the Athens Protocol (combined topography-guided partial PRK and CXL) in the right eye in February 2010 and in the left eye in September 2010. The treatment plan for both eyes was designed on the topography-guided wavelight excimer laser platform. RESULTS: Fifteen months after the right eye treatment, the right cornea had improved translucency and was topographically stable with uncorrected distance visual acuity (UDVA) 20/50 and CDVA 20/40 with refraction +0.50, -2.00 at 5°. We noted a similar outcome after similar treatment applied in the left eye with UDVA 20/50 and CDVA 20/40 with -0.50, -2.00 at 170° at the 8-month follow-up. CONCLUSION: In this case, the introduction of successful management of severe cornea abnormalities and scarring with the Athens Protocol may provide an effective alternative to other existing surgical or medical options.”
<Text 2>
“FISH studies were performed in 7 cases using the LSI BCR/ABL ES probe allowing the detection of the fusion BCR/ABL gene on the Ph chromosome in all of them and 9q34 deletions in 2 cases. Three cryptic complex rearrangements were detected by FISH studies. The third and the fourth chromosome regions involved in the 8 complex variant translocations were: 1q21, 1p36, 5q31, 11q13, 12q13, 12p13, and 20q12. In conclusion, FISH studies have been useful in the detection of the BCR/ABL rearrangements and 9q34 deletions, and to identify complex rearrangements that differ from the ones previously established by conventional cytogenetics.”
<Text 3>
“An alternative to killing? Treatment of reservoir hosts to control a vector and pathogen in a susceptible species. Parasite-mediated apparent competition occurs when one species affects another through the action of a shared parasite. One way of controlling the parasite in the more susceptible host is to manage the reservoir host. Culling can cause issues in terms of ethics and biodiversity impacts, therefore we ask: can treating, as compared to culling, a wildlife host protect a target species from the shared parasite? We used Susceptible Infected Recovered (SIR) models parameterized for the tick-borne louping ill virus (LIV) system. Deer are the key hosts of the vector (Ixodes ricinus) that transmits LIV to red grouse Lagopus lagopus scoticus, causing high mortality.”

***Question 1***:
Does the provided information confirm that the Athens Protocol was used as a treatment for the tick-borne louping ill virus in the 7 cases that underwent FISH studies? Yes, no, or maybe?
***Answer 1***: No.

***Question 2***:
The source texts indicate that the SIR models were parameterized for a 2010 study in which FISH analysis was used to treat cornea abnormalities in deer, the key hosts of the Ixodes ricinus vector. True, False, or Neither?
***Answer 2***:False.

***Question 3***:
Is it correct that a specific medical procedure performed in February 2010, a genetic analysis technique useful for detecting 9q34 deletions, and a wildlife management strategy for controlling Ixodes ricinus are all presented in the texts as alternatives to more drastic interventions? Correct, Incorrect, or Uncertain?
***Answer 3***:Incorrect.

**Examples End**
Please strictly follow the format of {***Question n***:} and ensure the questions must synthesize information from both source texts to be answered, adhering to all rules.
"""

question_prompt4 = """
**Instruction**:
The <Source Texts> block below contains four distinct and unrelated passages. Your task is to generate four questions that require synthesizing, comparing, or relating information from all four passages to be answered. Each question must be a closed-ended, logical inquiry that can be definitively answered based only on the provided text.
**Rules**:
Rules:
1. Synthesis Requirement: Each question must require the reader to process and integrate information from all four distinct texts provided. Questions that can be answered using only one of the texts are not acceptable.
2. Task Type: Three-Option Logical Inquiry.
3. Logical Inference: The question must be logically inferable from the input Source sentence(s). Do not use outside knowledge.
4. Question Suffix: The question must end with a three-option logical suffix. Choose one of the following formats:
Yes, no, or maybe?
True, False, or Neither?
Correct, Incorrect, or Uncertain?
5. Interrogative Structure: Use different interrogative structures (e.g., “Does…?”, “Is…?”, “Was…?”, “Have…?”, etc.) or frame the question around a statement.
6. Clarity and Grammar: The phrasing of the question should be clear, formal, and grammatically correct.

**Examples Start**: 
***Source Sentences***:
<Text 1>
“Cornea transplantation was the recommendation by multiple cornea specialists as the treatment of choice. We decided prior to considering a transplant to employ the Athens Protocol (combined topography-guided partial PRK and CXL) in the right eye in February 2010 and in the left eye in September 2010. The treatment plan for both eyes was designed on the topography-guided wavelight excimer laser platform. RESULTS: Fifteen months after the right eye treatment, the right cornea had improved translucency and was topographically stable with uncorrected distance visual acuity (UDVA) 20/50 and CDVA 20/40 with refraction +0.50, -2.00 at 5°. We noted a similar outcome after similar treatment applied in the left eye with UDVA 20/50 and CDVA 20/40 with -0.50, -2.00 at 170° at the 8-month follow-up. CONCLUSION: In this case, the introduction of successful management of severe cornea abnormalities and scarring with the Athens Protocol may provide an effective alternative to other existing surgical or medical options.”
<Text 2>
“FISH studies were performed in 7 cases using the LSI BCR/ABL ES probe allowing the detection of the fusion BCR/ABL gene on the Ph chromosome in all of them and 9q34 deletions in 2 cases. Three cryptic complex rearrangements were detected by FISH studies. The third and the fourth chromosome regions involved in the 8 complex variant translocations were: 1q21, 1p36, 5q31, 11q13, 12q13, 12p13, and 20q12. In conclusion, FISH studies have been useful in the detection of the BCR/ABL rearrangements and 9q34 deletions, and to identify complex rearrangements that differ from the ones previously established by conventional cytogenetics.”
<Text 3>
“An alternative to killing? Treatment of reservoir hosts to control a vector and pathogen in a susceptible species. Parasite-mediated apparent competition occurs when one species affects another through the action of a shared parasite. One way of controlling the parasite in the more susceptible host is to manage the reservoir host. Culling can cause issues in terms of ethics and biodiversity impacts, therefore we ask: can treating, as compared to culling, a wildlife host protect a target species from the shared parasite? We used Susceptible Infected Recovered (SIR) models parameterized for the tick-borne louping ill virus (LIV) system. Deer are the key hosts of the vector (Ixodes ricinus) that transmits LIV to red grouse Lagopus lagopus scoticus, causing high mortality.”
<Text 4>
“A Novel Deletion Mutation of SLC16A2 Encoding Monocarboxylate Transporter (MCT) 8 in a 26-year-old Japanese Patient with Allan-Herndon-Dudley Syndrome. Allan-Herndon-Dudley Syndrome (AHDS), an X linked condition, is characterized by congenital hypotonia that progresses to spasticity with severe psychomotor delays, in combination with altered thyroid hormone levels, in particular, high serum T3 levels.”

***Question 1***:
Is the following statement correct: The Athens Protocol, a treatment for corneal scarring, was applied to the 26-year-old patient with Allan-Herndon-Dudley Syndrome, and its success was confirmed using FISH studies to analyze the SIR models that govern the louping ill virus? Correct, Incorrect, or Uncertain?
***Answer 1***: Incorrect.

***Question 2***:
Does the information provided in the four texts suggest that treating a reservoir host like deer, a strategy analyzed with SIR models, is a therapeutic approach used to correct either the SLC16A2 deletion mutation found in AHDS or the BCR/ABL gene fusion detected via FISH studies, similar to how the Athens Protocol corrects corneal abnormalities? Yes, no, or maybe?
***Answer 2***:No.

***Question 3***:
The successful management of severe cornea abnormalities via the Athens Protocol was based on correcting a genetic deletion, such as the SLC16A2 mutation in the 26-year-old patient, which was identified using Susceptible Infected Recovered (SIR) models in the same way FISH studies were used to detect pathogens in red grouse. True, False, or Neither?
***Answer 3***:False.

**Examples End**
Please strictly follow the format of {***Question n***:} and ensure the questions must synthesize information from both source texts to be answered, adhering to all rules.
"""
