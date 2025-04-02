# Docker Setup

## Prerequisites

- Docker installed on your machine
- Docker Compose installed on your machine
- Data directory with the following structure:
  - Due to the large EEG data size, please save it outside the working dir

## Setup

Update the data path in `docker-compose.yml`:

```yaml
volumes:
  - .:/app
  - /path/to/your/data:/app/benchmark/Data:ro
```

Replace `/path/to/your/data` with the actual path to your data directory.
I only copied `/Data/EEG` dataset from the eos server because the entire dataset is too large.

## Running the Container

1. Build and start the container:

   ```bash
   docker compose up -d
   ```

2. Connect to the container shell:

   ```bash
   docker compose exec eeg_seizure bash
   ```

3. Activate the conda environment to load all the dependencies:

   ```bash
   conda activate eeg
   ```

4. Run your scripts:

   ```bash
   python benchmark/my_eeg_conformer/main.py
   ```

5. Exit the container shell:
   ```bash
   exit
   ```
   or press `Ctrl + D`

## Stopping the Container

To completely stop the container:

```bash
docker compose down
```

## Notes

- The container runs in CPU-only mode
- The data directory is mounted as read-only (`:ro` flag)
- Your local code changes will be reflected immediately in the container due to the volume mount
- The conda environment is pre-configured with all necessary dependencies

## Troubleshooting

If you encounter any issues:

1. Check if the container is running:

   ```bash
   docker compose ps
   ```

2. View container logs:

   ```bash
   docker compose logs
   ```

3. Rebuild the container if needed:
   ```bash
   docker compose down
   docker compose build --no-cache
   docker compose up -d
   ```
