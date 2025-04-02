# Use a minimal conda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment.yml
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate conda environment in shell
SHELL ["conda", "run", "-n", "eeg", "/bin/bash", "-c"]

# Default command
# CMD ["conda", "run", "-n", "eeg", "python", "benchmark/my_eeg_conformer/my_eeg_dataset.py"]