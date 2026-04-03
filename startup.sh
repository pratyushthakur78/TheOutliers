#!/usr/bin/env bash
set -e

# Azure App Service provides PORT dynamically.
exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8000}" --server.headless true
