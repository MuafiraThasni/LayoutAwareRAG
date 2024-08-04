from setuptools import setup, find_packages

setup(
    name='la-rag',
    version='0.1.4',
    description='Layout Aware RAG',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Muafira',
    author_email='muafirathasnikt@gmail.com',
    url='https://github.com/MuafiraThasni/LayoutAwareRAG',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your project dependencies here
        'nltk',
        'transformers',
        'torch'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    include_package_data=True,
)
