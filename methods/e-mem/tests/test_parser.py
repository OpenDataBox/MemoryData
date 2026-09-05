"""Tests for parser utilities."""

from src.utils.parser import limit_memory_segments


class TestLimitMemorySegments:
    """Test limit_memory_segments function."""

    def test_limit_segments_basic(self):
        """Test basic segment limiting."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
    <memory_segment>Memory 2</memory_segment>
    <memory_segment>Memory 3</memory_segment>
    <memory_segment>Memory 4</memory_segment>
    <memory_segment>Memory 5</memory_segment>
</relevant_memories>
<model_reasoning>Test reasoning</model_reasoning>"""

        result = limit_memory_segments(response, max_segments=3)

        # Should have exactly 3 segments
        assert result.count("<memory_segment>") == 3
        assert result.count("</memory_segment>") == 3
        assert "Memory 1" in result
        assert "Memory 2" in result
        assert "Memory 3" in result
        assert "Memory 4" not in result
        assert "Memory 5" not in result

    def test_no_limit_when_under_max(self):
        """Test that no limiting occurs when under max."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
    <memory_segment>Memory 2</memory_segment>
</relevant_memories>"""

        result = limit_memory_segments(response, max_segments=5)

        # Should remain unchanged
        assert result.count("<memory_segment>") == 2
        assert "Memory 1" in result
        assert "Memory 2" in result

    def test_non_retrieval_response_unchanged(self):
        """Test that non-retrieval responses are unchanged."""
        response = """<response_type>summary</response_type>
<summary_content>
    <speakers>Alice, Bob</speakers>
    <main_events>Discussion about project</main_events>
</summary_content>"""

        result = limit_memory_segments(response, max_segments=1)

        # Should be unchanged
        assert result == response

    def test_empty_response(self):
        """Test with empty response."""
        result = limit_memory_segments("", max_segments=5)
        assert result == ""

    def test_none_response(self):
        """Test with None response."""
        result = limit_memory_segments(None, max_segments=5)
        assert result is None

    def test_zero_max_segments(self):
        """Test with zero max_segments."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
</relevant_memories>"""

        result = limit_memory_segments(response, max_segments=0)
        assert result == response

    def test_negative_max_segments(self):
        """Test with negative max_segments."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
</relevant_memories>"""

        result = limit_memory_segments(response, max_segments=-1)
        assert result == response

    def test_multiline_segments(self):
        """Test with multiline segment content."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1
    Line 2
    Line 3</memory_segment>
    <memory_segment>Memory 2
    Another line</memory_segment>
    <memory_segment>Memory 3</memory_segment>
</relevant_memories>"""

        result = limit_memory_segments(response, max_segments=2)

        assert result.count("<memory_segment>") == 2
        assert "Memory 1" in result
        assert "Line 2" in result
        assert "Memory 2" in result
        assert "Memory 3" not in result

    def test_preserves_other_content(self):
        """Test that other content is preserved."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
    <memory_segment>Memory 2</memory_segment>
</relevant_memories>
<model_reasoning>
    Based on the segments above, the answer is: Test answer.
</model_reasoning>"""

        result = limit_memory_segments(response, max_segments=1)

        assert "<model_reasoning>" in result
        assert "Test answer" in result
        assert result.count("<memory_segment>") == 1

    def test_no_relevant_memories_tag(self):
        """Test response without relevant_memories tag."""
        response = """<response_type>retrieval</response_type>
<other_content>Some content</other_content>"""

        result = limit_memory_segments(response, max_segments=1)
        assert result == response

    def test_exact_max_segments(self):
        """Test when segments equal max_segments."""
        response = """<response_type>retrieval</response_type>
<relevant_memories>
    <memory_segment>Memory 1</memory_segment>
    <memory_segment>Memory 2</memory_segment>
    <memory_segment>Memory 3</memory_segment>
</relevant_memories>"""

        result = limit_memory_segments(response, max_segments=3)

        # Should be unchanged (or minimally changed for formatting)
        assert result.count("<memory_segment>") == 3

