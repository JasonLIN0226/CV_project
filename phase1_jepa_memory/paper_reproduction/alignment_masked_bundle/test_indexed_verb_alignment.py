import unittest
import numpy as np
from indexed_verb_alignment import selected_segments,indexed_multihot,labels_with_availability


class Poison(dict):
    def __getitem__(self,key):
        if key!='video':raise AssertionError('sealed numeric field read')
        return super().__getitem__(key)


class AlignmentTests(unittest.TestCase):
    def test_names_filtered_before_any_sealed_numeric_interpretation(self):
        row={'video':'approved/C10095_rgb.mp4','start_frame':'0','end_frame':'2','action_id':'3','verb_id':'1','noun_id':'2'}
        result=list(selected_segments([Poison(video='sealed/C10095_rgb.mp4'),Poison(video='approved/other.mp4'),row],{row['video']},{3:(1,2)}))
        self.assertEqual(result,[(row['video'],0,2,3,1,2)])
    def test_zero_origin_and_inclusive_end_follow_official_jpeg_axis(self):
        labels=indexed_multihot([1,2,3,4],[(0,2,1)])
        np.testing.assert_array_equal(labels[:,1],[1,1,1,0])
    def test_overlapping_same_verb_does_not_end_prematurely(self):
        labels=indexed_multihot(range(1,7),[(0,2,1),(1,4,1),(2,3,5)])
        np.testing.assert_array_equal(labels[:,1],[1,1,1,1,1,0])
        np.testing.assert_array_equal(labels[:,5],[0,0,1,1,0,0])
    def test_sparse_keys_use_actual_frame_index_not_sample_position(self):
        labels=indexed_multihot([1,16,31],[(15,15,7)])
        np.testing.assert_array_equal(labels[:,7],[0,1,0])
    def test_bad_key_axis_rejected(self):
        for frames in ([0,15],[1,1],[16,1]):
            with self.assertRaises(ValueError):indexed_multihot(frames,[(0,5,1)])
    def test_unknown_selected_label_does_not_become_background(self):
        row={'video':'a','start_frame':'0','end_frame':'2','action_id':'-1','verb_id':'-1','noun_id':'-1'}
        with self.assertRaises(ValueError):list(selected_segments([row],{'a'},{3:(1,2)}))
    def test_unlabelled_and_known_background_have_distinct_supervision_masks(self):
        unknown,unknown_mask=labels_with_availability([1,16,31],[],False)
        background,background_mask=labels_with_availability([1,16,31],[],True)
        np.testing.assert_array_equal(unknown,background)
        self.assertFalse(unknown_mask.any());self.assertTrue(background_mask.all())


if __name__=='__main__':unittest.main()
