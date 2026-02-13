#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from file_utils import read_document

def test_pdf_reading():
    # Test with a known PDF file
    test_pdf = r"../uploaded_docs/e075e60a/Two Friends.pdf"

    if os.path.exists(test_pdf):
        print(f"Testing PDF: {test_pdf}")
        print(f"File exists: {os.path.exists(test_pdf)}")
        print(f"File size: {os.path.getsize(test_pdf)} bytes")

        try:
            text = read_document(test_pdf)
            print(f"Text extracted successfully!")
            print(f"Text length: {len(text)} characters")
            print(f"First 300 characters:")
            print(repr(text[:300]))
            print("\n" + "="*50 + "\n")
            print(text[:500])  # Show actual text
        except Exception as e:
            print(f"Error extracting text: {e}")
    else:
        print(f"Test file not found: {test_pdf}")

if __name__ == "__main__":
    test_pdf_reading()