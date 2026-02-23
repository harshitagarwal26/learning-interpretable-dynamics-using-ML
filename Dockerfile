FROM python:3.8-slim-buster

WORKDIR /Lagrangian_caVAE-main


COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install jupyterlab 

# Fix outdated Debian Buster repositories
RUN sed -i 's|deb.debian.org|archive.debian.org|g' /etc/apt/sources.list && \
    sed -i '/security.debian.org/d' /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid

# Install only Xvfb (required for pyvirtualdisplay)
RUN apt-get update && apt-get install -y xvfb && rm -rf /var/lib/apt/lists/*

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

