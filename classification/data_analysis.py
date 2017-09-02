#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

CATEGORIES = [u'IS_COMPLAINT', u'IS_PRIVACY', u'IS_HARDWARE', u'IS_DEVICE', u'IS_PERFORMANCE', u'IS_BATTERY',
              u'IS_PRICE', u'IS_APP USABILITY', u'IS_ANDROID VERSION', u'IS_UI', u'IS_LICENSING', u'IS_MEMORY',
              u'IS_SECURITY']


NAME_PACKAGE_DICT = {
                    'Adblock Plus': 'org.adblockplus.adblockplussbrowser',
                    'Pixel Dungeon': 'com.watabou.pixeldungeon',
                    'Bubble level': 'com.gamma.bubblelevel',
                    'A Comic Viewer': 'net.androidcomics.acv',
                    'Muzei Live Wallpaper': 'net.nurik.roman.muzei',
                    'Signal Private Messenger': 'org.thoughtcrime.securesms',
                    'Amaze File Manager': 'com.amaze.filemanager',
                    'DashClock Widget': 'net.nurik.roman.dashclock',
                    'OctoDroid': 'com.gh4a',
                    'Device Control [root]': 'org.namelessrom.devicecontrol',
                    'AntennaPod': 'de.danoeh.antennapod',
                    'Xabber': 'com.xabber.android',
                    'MultiPicture Live Wallpaper': 'org.tamanegi.wallpaper.multipicture',
                    'Duck Duck GO': 'com.duckduckgo.mobile.android',
                    'Wally': 'me.wally.android.plus',
                    'QKSMS - Open Source SMS & MMS': 'com.moez.QKSMS',
                    'SeriesGuide': 'com.battlelancer.seriesguide',
                    'OS Monitor': 'com.eolwral.osmonitor',
                    'Simon Tatham\'s Puzzles': 'name.boyle.chris.sgtpuzzles',
                    'ConnectBot': 'org.connectbot',
                    'Terminal Emulator for Android': 'jackpal.androidterm',
                    'BatteryBot Battery Indicator': 'com.darshancomputing.BatteryIndicator',
                    'Network Log': 'com.googlecode.networklog',
                    'c:geo': 'cgeo.geocaching',
                    'Abstract Art': 'net.georgewhiteside.android.abstractart',
                    'Marine Compass': 'net.pierrox.mcompass',
                    'Turbo Editor ( Text Editor )': 'com.maskyn.fileeditor',
                    'Twidere for Twitter': 'org.mariotaku.twidere',
                    'Clip Stack ✓ Clipboard Manager': 'com.catchingnow.tinyclipboardmanager',
                    'Autostarts': 'com.elsdoerfer.android.autostarts',
                    'Financius - Expense Manager': 'com.code44.finance',
                    'Calculator': 'com.xlythe.calculator.material',
                    'Last.fm': 'fm.last.android',
                    'AcDisplay': 'com.achep.acdisplay',
                    'AnkiDroid Flashcards': 'com.ichi2.anki',
                    'CatLog': 'com.nolanlawson.logcat',
                    'Tweet Lanes': 'com.tweetlanes.android',
                    }


def print_category_data(data, category):
    category_data = data.loc[data[category] == 1]
    print("------------------------------------")
    for index, row in category_data.iterrows():
        print(row["reviewText"])
    print("------------------------------------")


category_words = ["donating", "donate", "donation", "$", "bucks", "free",
                  "lollipop", "marshmallow", "nougat", "kitkat",
                  "sd card", "sensor", "accelerometer", "camera", "sensors", "cpu",
                  "permission", "permissions", "privacy", "personal", "private", "track", "noninvasive", "invasive",
                  "memory error", "memory", "out of memory", "ram", "small", "low memory",
                  "kill battery", "drain", "consume", "battery friendly"
                  ]


def main():
    data = pd.read_csv("./data/train_reviews_manual_copy_04.csv")
    additional_data = pd.read_csv("./data/additional_data_02.csv")
    print("Total reviews: %d" % len(data))
    print(len(additional_data))
    ids = set(additional_data["_id"])
    print(len(ids))
    additional_data.drop_duplicates(subset=['_id'], inplace=True)
    print(len(additional_data))
    # complete_data.to_csv("./data/complete_data.csv", index=False)
    print(additional_data.columns)


if __name__ == "__main__":
    main()