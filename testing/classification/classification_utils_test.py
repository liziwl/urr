from __future__ import absolute_import
import unittest
from classification.classification_utils import *


class TestClassificationUtils(unittest.TestCase):

    def test_build_categories_list(self):
        expected_categories_list = ['IS_COMPLAINT', 'IS_SECURITY', 'IS_PRIVACY', 'IS_LICENSING', 'IS_PRICE',
                                    'IS_APP USABILITY', 'IS_UI', 'IS_DEVICE', 'IS_ANDROID VERSION', 'IS_HARDWARE',
                                    'IS_PERFORMANCE', 'IS_BATTERY', 'IS_MEMORY']
        expected_categories_list.sort()
        actual_categories_list = build_categories_list()
        actual_categories_list.sort()
        self.assertEqual(expected_categories_list, actual_categories_list)

    def test_build_pretty_categories_list(self):
        expected_pretty_categories_list = [('IS_COMPLAINT', 'Complaint'), ('IS_SECURITY', 'Security'),
                                           ('IS_PRIVACY', 'Privacy'), ('IS_LICENSING', 'Licensing'),
                                           ('IS_PRICE', 'Price'), ('IS_APP USABILITY', 'App Usability'),
                                           ('IS_UI', 'Ui'), ('IS_DEVICE', 'Device'),
                                           ('IS_ANDROID VERSION', 'Android Version'), ('IS_HARDWARE', 'Hardware'),
                                           ('IS_PERFORMANCE', 'Performance'), ('IS_BATTERY', 'Battery'),
                                           ('IS_MEMORY', 'Memory')]
        expected_pretty_categories_list.sort()
        actual_pretty_categories_list = build_pretty_categories_list()
        actual_pretty_categories_list.sort()
        self.assertEquals(expected_pretty_categories_list, actual_pretty_categories_list)

    def test_preprocess_review(self):
        expected_prep_review = preprocess_review("Cant activate lockscreen.... The app is not comes in usage access "
                                                 "permission option. Using Redmi Note 4G with Resurrection M Rom")
        actual_prep_review = "cant activ lockscreen the app is not come in usag access permiss option use redmi note " \
                             "4g with resurrect m rom"
        self.assertEquals(expected_prep_review, actual_prep_review)


if __name__ == '__main__':
    unittest.main()