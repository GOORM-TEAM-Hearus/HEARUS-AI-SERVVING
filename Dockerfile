FROM python:3.12.4

WORKDIR /src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd

# Copy application files
COPY main.py /src/
COPY routers /src/routers
COPY templates /src/templates
COPY images /src/images

# Create a shell script to activate venv and start the application
RUN echo '#!/bin/bash\n\
chmod 777 .venv/bin/activate\n\
source .venv/bin/activate\n\
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload\n\
' > /src/start_app.sh

RUN chmod +x /src/start_app.sh

CMD ["/src/start_app.sh"]