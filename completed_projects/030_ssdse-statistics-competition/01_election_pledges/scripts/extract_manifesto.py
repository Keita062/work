#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber

# Base paths
BASE_ELECTION_PATH = Path(__file__).parent / "election"
BASE_DATA_PATH = Path(__file__).parent / "data"

# Create output directories
(BASE_DATA_PATH / "衆議院").mkdir(parents=True, exist_ok=True)
(BASE_DATA_PATH / "参議院").mkdir(parents=True, exist_ok=True)

def extract_text_from_html(html_path):
    """Extract text from HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        print(f"Error extracting from HTML {html_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n---PAGE BREAK---\n"
        return text
    except Exception as e:
        print(f"Error extracting from PDF {pdf_path}: {e}")
        return None

def find_main_html_file(party_path):
    """Find the main HTML file (not in _files folder)"""
    for file in party_path.iterdir():
        if file.is_file() and file.suffix == '.html':
            # Exclude files in _files folders
            if '_files' not in str(file):
                return file
    return None

def find_pdf_file(party_path):
    """Find a PDF file in the party folder"""
    for file in party_path.iterdir():
        if file.is_file() and file.suffix == '.pdf':
            return file
    return None

def process_election_type(election_type, election_folder_name):
    """Process either 衆議院 (2025_shu) or 参議院 (2025_san)"""
    election_path = BASE_ELECTION_PATH / election_folder_name

    if not election_path.exists():
        print(f"Election path not found: {election_path}")
        return

    # Get all party folders
    for party_folder in sorted(election_path.iterdir()):
        if not party_folder.is_dir():
            continue

        party_name = party_folder.name
        print(f"Processing: {election_type} - {party_name}")

        # Try to find HTML file first
        html_file = find_main_html_file(party_folder)
        extracted_text = None

        if html_file:
            print(f"  Found HTML: {html_file.name}")
            extracted_text = extract_text_from_html(html_file)
        else:
            # Try PDF
            pdf_file = find_pdf_file(party_folder)
            if pdf_file:
                print(f"  Found PDF: {pdf_file.name}")
                extracted_text = extract_text_from_pdf(pdf_file)

        # Save extracted text
        if extracted_text:
            output_file = BASE_DATA_PATH / election_type / f"{party_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"  Saved to: {output_file}")
        else:
            print(f"  WARNING: Could not extract text for {party_name}")

# Process both election types
print("=" * 50)
print("Extracting 衆議院 (House of Representatives) manifestos...")
print("=" * 50)
process_election_type("衆議院", "2025_shu")

print("\n" + "=" * 50)
print("Extracting 参議院 (House of Councillors) manifestos...")
print("=" * 50)
process_election_type("参議院", "2025_san")

print("\n" + "=" * 50)
print("Extraction complete! ✓")
print("=" * 50)
