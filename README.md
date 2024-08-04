# Layout-Aware RAG (LA-RAG)

This is the source code repository for Python package [la-rag](https://pypi.org/project/la-rag/)

## Why Layout-Aware RAG?
The impressive abilities of large language models (LLMs) offer exciting possibilities for large-scale document analysis. However, a significant challenge remains in making text from extensive documents, such as large PDFs, accessible to LLMs due to their limited context window, which restricts the amount of text they can process at a time.

Retrieval Augmented Generation (RAG) systems address this challenge by combining LLMs with advanced retrieval techniques. Common chunking technologies used in LangChain, such as TextSplitter, RecursiveCharacterTextSplitter, etc break documents into smaller sections to fit within the LLM’s context window.
But, these methods will allow the system to lose the semantic connection of layout features, such as sections & subsections, tables, lists, bullet points, etc. For example, in a bullet-pointed list, all the points can be interrelated, and each point is connected to the paragraph or the last sentence of the paragraph given above the list.

ayout-aware RAG considers the layout features of the document and the semantic connection between them.

![alt text](<layout aware chunking demo.png>)

## Important Note
Although I have made the repository public and released the python package, I am still working on this project. I will be adding more details on the installation and building projects using la-rag soon. I encourage you to look into the tests\sample.py for reference to a sample code to see how to use the package. 

If you are interested to collaborate on this project, please email me at muafirathasnikt@gmail.com