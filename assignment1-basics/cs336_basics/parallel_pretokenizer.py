import multiprocessing as mp
import os
from typing import BinaryIO

from regex import W
from .tokenizer import Corpus, merge_corpora, pretokenize_to_corpus

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(input_path: str | os.PathLike, special_tokens: list[bytes], start: int, end: int) -> Corpus:
    with open(input_path, "rb") as f:
        f.seek(start)
        corpus = pretokenize_to_corpus(f.read(end - start), special_tokens)
        return corpus

def parallel_pretokenize_path_to_corpus(
        input_path: str | os.PathLike,
        special_tokens: list[str] = ["<|endoftext|>"],
        processes: int = mp.cpu_count(),
    ):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, processes, "<|endoftext|>".encode("utf-8"))

        special_tokens_bytes = [t.encode('utf-8') for t in special_tokens]

        jobs = [(input_path, special_tokens_bytes, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

        results: list[Corpus] = []
        with mp.Pool(processes=processes) as pool:
            results = pool.starmap(process_chunk, jobs)

        corpus = merge_corpora(*results)
        return corpus
