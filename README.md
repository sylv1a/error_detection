AUTOMATED ERROR DETECTION IN RADIOLOGY REPORTS WITH LLMS

Description
Medical findings on X-ray images are typically documented in free-text radiology
reports. These reports are crucial in making critical clinical decisions.
However, these reports can have errors and inconsistencies that negatively
impact patient care and further treatment. Although large language models
(LLMs) can automatically detect errors in radiology reports, their reliance
on cloud-based infrastructure is a serious concern with regard to data privacy.
Our work explores the potential of locally hosted open source LLMs
for automated inconsistency checks and error detection in radiology reports.
In our approach, we use advanced prompting techniques, such as Chain of
Thought (CoT) and Prompt Chaining, to detect internal contradictions, as
well as standalone errors such as date and measurement errors. We also take
into account the accuracy of the error positions in our result, which has been
missing from previous studies. We propose a robust method to detect contradictions within finding and impression of a
report, as well as stand-alone errors.

Usage
To utilize this tool, you can run app.py file in which you will be asked to provide report in which you want to perform detection

Authors and Acknowledgement
This project is collaborative effort by :

Prof. Dr. Michael Ingrisch, Dr. Andreas Mittermeier and Dr. Philipp Wesp
Sameer Singh Rawat and Sylvia Chacha (LMU)
