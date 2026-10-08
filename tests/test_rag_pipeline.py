import unittest

from src.rag_pipeline import chunk_text


class TestRAGPipeline(unittest.TestCase):

    def test_chunk_text_creates_chunks(self):
        text = "A" * 1200

        chunks = chunk_text(
            text,
            chunk_size=500,
            overlap=100
        )

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 500)

    def test_chunk_overlap(self):
        text = "".join(str(i % 10) for i in range(1000))

        chunks = chunk_text(
            text,
            chunk_size=500,
            overlap=100
        )

        self.assertEqual(
            chunks[0][-100:],
            chunks[1][:100]
        )

    def test_empty_text(self):
        chunks = chunk_text("")

        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
