import unittest

from rag.retriever import _is_duplicate_chunk, _postprocess_results


class RetrieverFilterTests(unittest.TestCase):
    def test_exact_duplicate_is_removed(self):
        first = {
            "source": "doc.txt",
            "chunk_index": 0,
            "word_start": 0,
            "word_end": 10,
            "section_title": "Section",
            "text": "The same evidence appears here.",
        }
        duplicate = dict(first)
        duplicate["chunk_index"] = 1
        self.assertTrue(_is_duplicate_chunk(duplicate, [first]))

    def test_per_source_cap_is_applied(self):
        chunks = [
            {
                "source": "a.txt",
                "chunk_index": index,
                "word_start": index * 10,
                "word_end": index * 10 + 5,
                "section_title": f"Section {index}",
                "text": f"Unique evidence {index}",
            }
            for index in range(4)
        ]
        chunks.append({
            "source": "b.txt",
            "chunk_index": 0,
            "word_start": 0,
            "word_end": 5,
            "section_title": "Other",
            "text": "Other unique evidence",
        })
        filtered = _postprocess_results(chunks, 5, comprehensive=False)
        self.assertLessEqual(sum(item["source"] == "a.txt" for item in filtered), 3)


if __name__ == "__main__":
    unittest.main()
