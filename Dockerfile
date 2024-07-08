FROM public.ecr.aws/lambda/python:3.9.2024.06.10.11

RUN pip install langchain google.generativeai langchain-google-genai langchain-text-splitters youtube-transcript-api langchain-community pinecone-client requests nltk inflect langchain-experimental boto3 pandas tabulate

COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.handler" ]
