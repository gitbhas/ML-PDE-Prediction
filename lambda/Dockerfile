FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r requirements.txt

# Copy function code and models
COPY . ${LAMBDA_TASK_ROOT}

CMD ["prediction_handler.lambda_handler"]