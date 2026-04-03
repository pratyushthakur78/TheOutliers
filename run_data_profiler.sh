#!/usr/bin/env bash
# Run from Git Bash on Windows, or Linux/macOS if paths match.
set -e
cd "/d/OneDrive - Biz2X Platform Pvt Ltd/Documents/Hackathon" 2>/dev/null || \
  cd "D:/OneDrive - Biz2X Platform Pvt Ltd/Documents/Hackathon"
echo "Starting Streamlit..."
py -3 -m streamlit run app.py
