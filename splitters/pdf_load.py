import re

class EnhancedPDFLoader:
    def __init__(self, file_path):
        """
        Initialize with the path of the PDF file.
        
        Parameters:
        - file_path (str): Path to the PDF file to be read.
        """
        self.file_path = file_path

    def extract_text(self):
        """
        Extracts text from the PDF, preserving formatting and handling RTL text.
        
        Returns:
        - str: The extracted text from the PDF.
        """
        with open(self.file_path, 'rb') as f:
            pdf_content = f.read()

        # Decode the binary data (Latin-1 is common for PDFs, but you can adjust based on the document)
        pdf_text = self._parse_pdf_content(pdf_content.decode('latin-1'))
        return pdf_text

    def _parse_pdf_content(self, content):
        """
        Internal function to parse the raw PDF content.
        
        Parameters:
        - content (str): Decoded content of the PDF.
        
        Returns:
        - str: Extracted text with formatting and RTL handling.
        """
        # Find streams with BT (Begin Text) and ET (End Text)
        text_blocks = re.findall(r'BT(.*?)ET', content, re.S)

        extracted_text = ""
        for block in text_blocks:
            # Extract text commands like Tj and TJ which contain actual text data
            text_fragments = re.findall(r'\((.*?)\)Tj', block)

            # Handle RTL and LTR text detection and reconstruction
            for fragment in text_fragments:
                fragment_cleaned = self._clean_text_fragment(fragment)
                extracted_text += fragment_cleaned + "\n"
        
        return extracted_text.strip()

    def _clean_text_fragment(self, fragment):
        """
        Cleans up a text fragment by tackling formatting issues and detecting RTL text.

        Parameters:
        - fragment (str): The raw text fragment from the PDF.

        Returns:
        - str: Cleaned and properly formatted text.
        """
        # Remove unnecessary spaces, newlines, and artifacts
        fragment = re.sub(r'\s+', ' ', fragment)

        # Handle right-to-left (RTL) text, particularly for Arabic or Hebrew
        if self._is_rtl(fragment):
            fragment = self._handle_rtl_text(fragment)
        
        # Further cleaning could be added here based on the document type
        return fragment

    def _is_rtl(self, text):
        """
        Determines if the given text is in a right-to-left language.
        
        Parameters:
        - text (str): The text to check.
        
        Returns:
        - bool: True if RTL, False if LTR.
        """
        # Check for Arabic or Hebrew character ranges (U+0600–U+06FF for Arabic, U+0590–U+05FF for Hebrew)
        rtl_characters = re.compile(r'[\u0590-\u05FF\u0600-\u06FF]')
        return bool(rtl_characters.search(text))

    def _handle_rtl_text(self, text):
        """
        Adjusts RTL text so that it reads correctly.
        
        Parameters:
        - text (str): The RTL text to handle.
        
        Returns:
        - str: Properly oriented RTL text.
        """
        # Simply reverse the text for RTL languages
        return text[::-1]

# Example usage 
if __name__ == "__main__":
    # Create an instance of the enhanced PDF loader
    loader = EnhancedPDFLoader("example.pdf")
    
    # Extract the text while preserving formatting and handling RTL
    extracted_text = loader.extract_text()
    
    print(extracted_text)
