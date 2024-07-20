#%%
import sys
sys.path.append(r"C:\Users\User\Desktop\LayoutAwareRAG\src\la_rag")
import pymupdf
from pipeline import extractInfo as model
#%%
path = r"C:\Users\User\Desktop\personal\resume_pdf_sample.pdf"
name = "Resume"

result = model([path],[name],"experience")
# %%
