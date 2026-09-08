from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parent))
from crossview_probes import assign_views,fingerprint,require_fit,assess_view


class CrossviewTests(unittest.TestCase):
    def test_audit_views_cannot_enter_fit_or_memory_evaluation_before_lock(self):
        with self.assertRaises(ValueError):
            require_fit([{'view':'audit','split':'probe_view_audit'}])
        with self.assertRaises(ValueError):
            assess_view(None,{}, {},[{'view':'audit'}],'audit',Path('.'))

    def test_question_views_cannot_copy_any_cue_or_each_other(self):
        forbidden={fingerprint('Where did Alice go?')}
        with self.assertRaises(ValueError):
            assign_views('p',{'question_1':'Where did Alice go?','question_2':'Which place did she visit?'},forbidden)
        with self.assertRaises(ValueError):
            assign_views('p',{'question_1':'Which place?','question_2':'which place!'},forbidden)
        with self.assertRaises(ValueError):
            assign_views('p',{'question_1':'One?'},forbidden)

    def test_view_assignment_depends_only_on_id_not_question_content(self):
        a=assign_views('p',{'question_1':'First question?','question_2':'Second question?'},set())
        b=assign_views('p',{'question_1':'Different first?','question_2':'Different second?'},set())
        self.assertEqual(a['fit'].startswith('First'),b['fit'].endswith('first?'))
        self.assertNotEqual(a['fit'],a['audit'])


if __name__=='__main__':
    unittest.main()
