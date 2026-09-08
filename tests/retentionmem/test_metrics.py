import pytest
from methods.retentionmem.metrics import locomo_score


def test_locomo_category_specific_protocol():
    assert locomo_score("run", "running", 4) == 1
    assert locomo_score("blue", "blue; red is also possible", 3) == 1
    assert locomo_score("red, blue", "red, blue", 1) == 1
    assert locomo_score("red", "red, blue", 1) == .5
    assert locomo_score("Not mentioned in our history", "anything", 5) == 1
    assert locomo_score("", "", 4) == 0
    assert locomo_score("coffee and tea", "coffee tea", 2) == 1
    with pytest.raises(ValueError): locomo_score("a", "a", 6)
