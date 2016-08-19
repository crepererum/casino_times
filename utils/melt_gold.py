#!/usr/bin/env python3

import sys

import Levenshtein


ref = {
    "democraci": {
        0: [171269, 171274, 38662, 554250, 249870, 175395, 711479, 307228, 781509, 422506, 411793, 333965, 140904, 150376, 220268, 247240, 544825, 589162, 334229, 249823],
        5: [171269, 278371, 7784, 253071, 171274, 217133, 740440, 130421, 334401, 668146, 481418, 38666, 189122, 337398, 642453, 730604, 523933, 680174, 597131, 451612],
        10: [171269, 171274, 38666, 278371, 7784, 642453, 297634, 253071, 334401, 556966, 197910, 32956, 189122, 668146, 132463, 217133, 597131, 52726, 176354, 739257],
        15: [171269, 7784, 324955, 253071, 38666, 297634, 699813, 171274, 633970, 278371, 642453, 197910, 120384, 473602, 668146, 681624, 556966, 769509, 140124, 697362],
        20: [171269, 7784, 123832, 38666, 324955, 158952, 197910, 253071, 297634, 699813, 171274, 681624, 633970, 278371, 556966, 392929, 590998, 743899, 473602, 642453]
    },
    "drug": {
        0: [193656, 724553, 244668, 484754, 614870, 212437, 565389, 2260, 746986, 641499, 242993, 412501, 305197, 50963, 161445, 245374, 655719, 595533, 118895, 713881],
        5: [193656, 327512, 724553, 626048, 722078, 569769, 641699, 142419, 742229, 171459, 442449, 202130, 149795, 565270, 106586, 544841, 5811, 570414, 133042, 338191],
        10: [193656, 724553, 327512, 254024, 626048, 722078, 142419, 742229, 171459, 775581, 400881, 435496, 641699, 569769, 442449, 81147, 5811, 424473, 149795, 246240],
        15: [193656, 724553, 254024, 327512, 442449, 171459, 626048, 593062, 722078, 435496, 64303, 157954, 168853, 142419, 775581, 742229, 81147, 202130, 400881, 149795],
        20: [193656, 724553, 254024, 327512, 442449, 171459, 626048, 593062, 722078, 435496, 64303, 157954, 168853, 424473, 142419, 775581, 742229, 81147, 202130, 400881]
    },
    "german": {
        0: [265067, 265080, 770257, 251812, 704442, 759174, 451977, 223228, 210042, 176354, 342511, 39267, 532570, 661509, 247020, 81180, 19097, 142864, 338077, 327705],
        5: [265067, 265080, 770257, 451977, 19097, 39267, 251812, 661509, 223228, 342511, 226662, 704442, 330680, 728129, 19088, 327625, 723650, 596405, 759174, 248707],
        10: [265067, 265080, 770257, 451977, 19097, 251812, 39267, 330680, 661509, 223228, 342511, 768186, 226662, 327625, 728129, 130122, 394537, 704442, 337771, 110010],
        15: [265067, 265080, 770257, 451977, 550807, 661509, 206057, 19097, 704442, 683109, 251812, 768186, 330680, 10542, 39267, 772690, 110010, 223228, 226662, 342511],
        20: [265067, 265080, 683109, 770257, 550807, 451977, 10542, 661509, 704442, 110010, 650251, 206057, 337771, 19097, 251812, 768186, 330680, 141797, 328695, 58002]
    },
    "happi": {
        0: [292880, 452881, 550719, 496152, 425633, 484634, 224212, 326489, 21327, 224144, 490129, 519831, 719021, 495998, 494837, 481800, 781493, 174439, 761941, 212382],
        5: [292880, 496152, 452881, 484634, 550719, 224144, 425633, 550679, 494837, 374425, 481800, 776346, 224212, 519831, 761941, 329728, 455407, 246057, 174439, 492248],
        10: [292880, 496152, 452881, 484634, 550719, 224144, 425633, 550679, 494837, 481800, 374425, 519831, 776346, 761941, 224212, 329728, 492248, 455407, 454846, 246057],
        15: [292880, 496152, 452881, 484634, 550719, 224144, 425633, 550679, 494837, 481800, 374425, 519831, 776346, 761941, 224212, 329728, 492248, 455407, 454846, 246057],
        20: [292880, 496152, 452881, 484634, 550719, 224144, 425633, 550679, 494837, 481800, 374425, 519831, 776346, 761941, 224212, 329728, 492248, 455407, 454846, 246057]
    },
    "health": {
        0: [297411, 108332, 71483, 50963, 131718, 245374, 3659, 505377, 364199, 563705, 394810, 747556, 774059, 325741, 144425, 776623, 72131, 529898, 19417, 186992],
        5: [297411, 331528, 108332, 245374, 590741, 19417, 563705, 724533, 71483, 394810, 50963, 662414, 224891, 571831, 131718, 240544, 253659, 501265, 505377, 406411],
        10: [297411, 331528, 245374, 108332, 71483, 590741, 563705, 19417, 394810, 662414, 724533, 253659, 501769, 224891, 131718, 50963, 240544, 774059, 406411, 634426],
        15: [297411, 331528, 245374, 108332, 71483, 590741, 19417, 563705, 394810, 662414, 724533, 253659, 224891, 501769, 131718, 50963, 240544, 774059, 406411, 747556],
        20: [297411, 331528, 245374, 108332, 71483, 590741, 19417, 563705, 394810, 662414, 724533, 253659, 224891, 501769, 131718, 50963, 240544, 774059, 406411, 747556]
    },
    "know": {
        0: [374425, 186992, 495998, 484634, 776346, 709683, 314203, 517869, 490129, 297491, 719021, 659769,  99808, 706825,  41565,  51341, 403033, 191342, 295997, 509415],
        5: [374425, 186992, 484634, 495998, 776346, 517869, 314203, 191342, 490129, 709683, 519831, 51341, 455407, 706825, 297491, 41565, 659769, 481800, 509415, 719021],
        10: [374425, 186992, 484634, 495998, 776346, 517869, 314203, 191342, 490129, 709683, 519831, 51341, 455407, 706825, 297491, 41565, 481800, 659769, 509415, 719021],
        15: [374425, 186992, 484634, 495998, 776346, 517869, 314203, 191342, 490129, 709683, 519831, 51341, 455407, 706825, 481800, 297491, 41565, 659769, 509415, 719021],
        20: [374425, 186992, 484634, 495998, 776346, 517869, 314203, 191342, 490129, 709683, 519831, 51341, 455407, 706825, 481800, 297491, 41565, 659769, 509415, 719021]
    },
    "money": {
        0: [460082, 531776, 335525, 584446, 685215, 8627, 245730, 634097, 170851, 205510, 683389, 498609, 567360, 210792, 750391, 531873, 327990, 781493, 1948, 72254],
        5: [460082, 531776, 685215, 335525, 170851, 584446, 634097, 210792, 329642, 145534, 245730, 512314, 480665, 570414, 143183, 580763, 592798, 205510, 683986, 143302],
        10: [460082, 531776, 685215, 335525, 170851, 584446, 634097, 329642, 210792, 148855, 145534, 245730, 143310, 498609, 580763, 512314, 480665, 750391, 570414, 143183],
        15: [460082, 531776, 685215, 335525, 170851, 584446, 634097, 329642, 210792, 148855, 229937, 739261, 143310, 145534, 293188, 480657, 245730, 498609, 580763, 512314],
        20: [460082, 531776, 685215, 335525, 170851, 584446, 634097, 329642, 210792, 148855, 229937, 739261, 143310, 145534, 293188, 480657, 245730, 498609, 580763, 512314]
    },
    "religion": {
        0: [594893, 594870, 568778, 600079, 564105, 187408, 184543, 546177, 324173, 141624, 615813, 263918, 478414, 141758, 590785, 327208, 171891, 49291, 761941, 784172],
        5: [594893, 600079, 568778, 333118, 184543, 575795, 594870, 49291, 171891, 589626, 402323, 320225, 564105, 539429, 324173, 394750, 138201, 743389, 183639, 263918],
        10: [594893, 600079, 568778, 333118, 575795, 184543, 594870, 171891, 49291, 589626, 320225, 744856, 402323, 564105, 539429, 743389, 263918, 324173, 138201, 394750],
        15: [594893, 600079, 568778, 333118, 575795, 184543, 594870, 171891, 49291, 589626, 320225, 744856, 402323, 564105, 539429, 743389, 263918, 324173, 138201, 394750],
        20: [594893, 600079, 568778, 333118, 575795, 184543, 594870, 171891, 49291, 589626, 320225, 744856, 402323, 564105, 539429, 743389, 263918, 324173, 138201, 394750]
    },
    "soft": {
        0: [660881, 263845, 411066, 108332, 770554, 732585, 776623, 410934, 168070, 580959, 707869, 111506, 364199, 403033, 71483, 239095, 774059, 370275, 719021, 406130],
        5: [660881, 111506, 263845, 51341, 411066, 779316, 403033, 732585, 165973, 402569, 707869, 108332, 776623, 708493, 725597, 364199, 168070, 580959, 199341, 313483],
        10: [660881, 111506, 263845, 51341, 779316, 411066, 403033, 165973, 732585, 707869, 402569, 108332, 402323, 708493, 301899, 776623, 725597, 364199, 168070, 471394],
        15: [660881, 111506, 263845, 51341, 779316, 411066, 403033, 165973, 732585, 402569, 707869, 108332, 402323, 708493, 301899, 776623, 725597, 364199, 168070, 471394],
        20: [660881, 111506, 263845, 51341, 779316, 411066, 403033, 165973, 732585, 402569, 707869, 108332, 402323, 708493, 301899, 776623, 725597, 364199, 168070, 471394]
    },
    "war": {
        0: [770257, 532570, 211580, 237942, 46255, 451977, 39267, 39071, 138418, 142864, 661509, 8612, 62107, 337724, 245478, 428933, 223228, 19097, 150158, 599140],
        5: [770257, 237942, 532570, 249870, 211580, 451977, 327625, 478134, 673720, 46255, 62107, 711099, 585042, 150158, 39267, 245855, 39071, 175395, 245478, 535975],
        10: [770257, 237942, 532570, 249870, 211580, 327625, 673720, 451977, 478134, 535975, 150158, 46255, 39071, 711099, 585042, 62107, 687164, 761445, 245855, 518493],
        15: [770257, 237942, 532570, 249870, 673720, 211580, 327625, 451977, 711099, 39071, 142864, 478134, 761445, 579875, 242264, 585042, 150158, 683244, 535975, 687164],
        20: [770257, 237942, 532570, 249870, 673720, 211580, 327625, 39071, 451977, 711099, 142864, 478134, 761445, 579875, 242264, 585042, 150158, 683244, 535975, 245855]
    }
}


def compare_string(lref, lnew):
    limit = max(len(lref), len(lnew))
    fake_id = 0
    while len(lnew) < limit:
        lnew.append(fake_id)
        fake_id += 1

    counter = 0
    known = {}
    for entry in lref:
        if entry not in known:
            known[entry] = counter
            counter += 1

    newentries = 0
    for entry in lnew:
        if entry not in known:
            known[entry] = counter
            counter += 1
            newentries += 1

    string1 = ''.join(chr(known[x] + 1) for x in lref)
    string2 = ''.join(chr(known[x] + 1) for x in lnew)

    sdiff = 1 - Levenshtein.distance(string1, string2) / limit
    enew = newentries / limit

    return sdiff, enew


def f2s(f):
    return "{:.5f}".format(f)


fname_in = sys.argv[1]
fname_out = sys.argv[2]
fp_in = open(fname_in)
fp_out = open(fname_out, "w")

fp_out.write("maxerror,ngram,minweight,r,ndtw,sdiff,enew\n")

s = ""

try:
    while True:
        # search next part
        while not s.strip().startswith("===="):
            s = next(fp_in)

        # header
        maxerror, ngram, maxlevel, r = next(fp_in).strip().split(",")

        # open index file message
        next(fp_in)
        next(fp_in)

        # #dtw status
        _, ndtw = next(fp_in).strip().split("=")

        # table header
        next(fp_in)
        next(fp_in)

        # table content
        s = next(fp_in)
        ids = []
        try:
            while s.strip().startswith("|"):
                _, c_ngram, c_id, c_dist, _ = s.strip().split("|")
                ids.append(int(c_id.strip()))
                s = next(fp_in)
        except StopIteration:
            raise
        finally:
            sdiff, enew = compare_string(ref[ngram][int(r)], ids)
            fp_out.write(",".join([maxerror, ngram, maxlevel, r, ndtw, f2s(sdiff), f2s(enew)]))
            fp_out.write("\n")
except StopIteration:
    print("done")