"""Checks for the leak-prevention logic in models/yolo_caries/scripts/Junwei/dedup_and_split.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "yolo_caries" / "scripts" / "Junwei"))
from dedup_and_split import (adopt_loners, assign_splits, has_bad_box, patient_id,  # noqa: E402
                             person_groups, write_split)

T = 1_729_000_000_000  # a millisecond timestamp
MIN = 60_000


def rec(pid_part: str, session: str, ts: int) -> dict:
    key = f"no_retractors/Frontal/anonymous_{pid_part}-{session}_{ts}_Frontal_View"
    return {"key": key, "patient_id": patient_id(key.split("/")[-1]), "is_pilot": False}


def test_patient_id_dash_underscore_pilot():
    assert patient_id("anonymous_003-007-1214-00_1732862917124_Frontal_View") == "003-007-1214"
    assert patient_id("anonymous_003_007_673_00_1729239709286_Frontal_View") == "003-007-673"
    assert patient_id("anonymous-frontalView-1726821735681") is None


def test_two_sessions_of_one_person_are_one_group():
    records = [
        rec("003-103-297", "01", T),              # session 01
        rec("003-103-298", "00", T + 13 * MIN),   # same person, session 00, 13 min later
        rec("003-103-400", "00", T + 5 * MIN),    # number too far away: someone else
        rec("003-104-298", "00", T + 13 * MIN),   # different prefix: someone else
        rec("003-103-299", "01", T + 2 * 60 * MIN),  # two hours later: someone else
    ]
    g = person_groups(records)
    assert g["003-103-297"] == g["003-103-298"]
    assert len({g["003-103-297"], g["003-103-400"], g["003-104-298"], g["003-103-299"]}) == 4


def test_no_person_in_two_splits_and_pilot_in_train():
    records = []
    for n in range(0, 40, 2):  # 20 people, each with two IDs
        records += [rec(f"003-007-{n}", "01", T + n * 60 * MIN), rec(f"003-007-{n + 1}", "00", T + n * 60 * MIN + 10 * MIN)]
    records.append({"key": "pilot/Frontal/anonymous-frontalView-1726821735681", "patient_id": None, "is_pilot": True})
    assign_splits(records, [0.7, 0.15, 0.15], seed=42)
    splits_per_group = {}
    for r in records:
        splits_per_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for g, s in splits_per_group.items() if g)
    assert records[-1]["split"] == "train"


def test_loner_joins_nearest_group_and_split():
    records = [
        rec("003-103-297", "01", T), rec("003-103-298", "00", T + 13 * MIN),  # a normal pair
        rec("003-103-301", "01", T + 50 * MIN),      # number 3 away and 37 min after 298: missed by the strict rule
        rec("003-103-350", "01", T + 10 * 60 * MIN),  # ten hours later: stays alone
    ]
    strict = person_groups(records)
    assert strict["003-103-301"] != strict["003-103-298"]
    g = adopt_loners(records, strict)
    assert g["003-103-301"] == g["003-103-298"] == g["003-103-297"]
    assert g["003-103-350"] == "003-103-350"
    assign_splits(records, [0.7, 0.15, 0.15], seed=1)
    assert len({r["split"] for r in records[:3]}) == 1


def test_anyone_photographed_during_the_pilot_goes_to_train():
    for seed in range(20):  # whatever the shuffle does
        records = [rec(f"003-007-{n}", "01", T + n * 5 * 60 * MIN) for n in range(10)]
        records.append({"key": f"pilot/Frontal/anonymous-frontalView-{T + 6 * 60 * MIN}", "patient_id": None, "is_pilot": True})
        assign_splits(records, [0.4, 0.3, 0.3], seed=seed)
        assert records[0]["split"] == records[1]["split"] == "train"  # photographed before the last pilot photo


def test_bad_boxes_are_found_and_classes_become_zero(tmp_path):
    ok, bad = tmp_path / "ok.txt", tmp_path / "bad.txt"
    ok.write_text("1 0.5 0.5 0.2 0.2\n0 0.955 0.5 0.1 0.1\n")  # second box ends at 1.005: inside the 1% tolerance
    bad.write_text("0 0.99 0.5 0.1 0.1\n")                       # ends at 1.04
    assert not has_bad_box(ok) and has_bad_box(bad)

    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    records = [{"key": "retractors/Frontal/a", "image_path": img, "label_path": ok,
                "patient_id": "1-1-1", "group": "1-1-1", "split": "val"}]
    write_split(records, tmp_path / "out", tmp_path / "m.csv")
    assert (tmp_path / "out/labels/val/retractors__Frontal__a.txt").read_text() == "0 0.5 0.5 0.2 0.2\n0 0.955 0.5 0.1 0.1\n"
    assert (tmp_path / "out/images/val/retractors__Frontal__a.jpg").exists()
    assert "retractors/Frontal/a,1-1-1,1-1-1,val" in (tmp_path / "m.csv").read_text()


def test_distinct_skips_lookalike_photos(tmp_path):
    from PIL import Image, ImageDraw
    from build_mixed import distinct

    def photo(name, draw_it, quality=95):
        im = Image.new("RGB", (256, 256), "white")
        draw_it(ImageDraw.Draw(im))
        im.save(tmp_path / name, quality=quality)
        return tmp_path / name

    a = photo("a.jpg", lambda d: d.ellipse((40, 60, 200, 180), fill="black"))
    a_resaved = photo("a2.jpg", lambda d: d.ellipse((40, 60, 200, 180), fill="black"), quality=40)
    b = photo("b.jpg", lambda d: [d.rectangle((x, 0, x + 16, 256), fill="black") for x in range(0, 256, 48)])
    assert distinct([a, a_resaved, b], []) == [a, b]
    assert distinct([a, a_resaved, b], [], want=1) == [a]
