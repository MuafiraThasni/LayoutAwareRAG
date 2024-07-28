#%%
# import sys
# sys.path.append(r"C:\Users\User\Desktop\LayoutAwareRAG\src\la_rag")
# import pymupdf
from la_rag.pipeline import extractInfo as model

#%%
path = r"C:\Users\User\Desktop\Data Science Matrials\Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"
name = "ml book"

result = model([path],[name],"experience")
# %%
