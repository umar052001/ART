class SimpleTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=64):
        """
        Initialize with the desired chunk size and overlap.
        
        Parameters:
        - chunk_size (int): The maximum size of each chunk (default: 512 characters).
        - chunk_overlap (int): Number of overlapping characters between chunks (default: 64).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        """
        Splits the input text into chunks with overlap.
        
        Parameters:
        - text (str): The text to split into chunks.
        
        Returns:
        - list: A list of text chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks
