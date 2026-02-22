from quantum_llm.data import CharTokenizer


def test_char_tokenizer_roundtrip() -> None:
    text = "abc cab"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    out = tok.decode(ids)
    assert out == text
