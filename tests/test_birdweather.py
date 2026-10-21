"""What the frame reads off a BirdWeather station.

The same invariant as BirdNET-Go's at the top: a station that will not answer
raises, and only one that answers with nothing reads as no birds. Then the ways
BirdWeather differs: a station it does not have, days that are UTC's, and a
record too long to ask for on every tick.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import UTC, datetime, time, timedelta
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pytest

from fugleramme import birdweather, fake
from fugleramme.api import ApiSource, Configured
from fugleramme.birdweather import BirdWeatherSource
from fugleramme.settings import Settings, SettingsStore
from fugleramme.source import BIRDNET_GO, BIRDWEATHER, NoSuchStation, Unavailable

BLACKBIRD, TIT, ROBIN, CROW = "Turdus merula", "Parus major", "Erithacus rubecula", "Corvus cornix"
BAT = "Myotis daubentonii"


def _row(id_: int, at: datetime, name: str, station: str = fake.STATION_ID) -> fake.Detection:
    return fake.Detection(id_, at, name, 0.9, False, station)


def _page(url: str, *stations: str) -> str:
    return ", ".join(f"{url}/stations/{station}" for station in stations or [fake.STATION_ID])


@pytest.fixture
def station(detector):
    """A BirdWeatherSource over fresh fake stations, the first alone unless
    `stations` says otherwise; `rows` replaces the generated ones."""

    def start(
        rows=None, count: int = 40, seed: int = 1, stations=(fake.STATION_ID,), **kwargs
    ) -> BirdWeatherSource:
        if rows is None:
            rows = fake.generate(count, seed, fake.BIRDWEATHER_HEARD)
        url, _httpd = detector(rows=rows, birdweather=True, **kwargs)
        return BirdWeatherSource(_page(url, *stations))

    return start


def _asked(source: BirdWeatherSource) -> list:
    return [
        source.latest,
        source.life_list,
        source.stats,
        lambda: source.recent(5),
        lambda: source.species_since(6),
        lambda: source.species_since(0),
        lambda: source.last_heard(24),
    ]


def test_a_station_that_will_not_answer_raises(station):
    for call in _asked(station(down=True)):
        with pytest.raises(Unavailable) as raised:
            call()
        assert not isinstance(raised.value, NoSuchStation)


def test_a_station_birdweather_does_not_have_is_its_own_failure(detector):
    """BirdWeather answers a station it does not have with a 404 page. Called
    unreachable, it would send the reader to check a network that is fine."""
    url, _httpd = detector(birdweather=True)
    for source in (BirdWeatherSource(_page(url, "999")), BirdWeatherSource(url)):
        for call in _asked(source):
            with pytest.raises(NoSuchStation):
                call()


@pytest.mark.parametrize(
    "answer",
    [
        {"errors": [{"message": "Query requests 4 detection fields (limit 3)."}]},
        {"data": None},
        {"data": {}},
        {"data": {"station": None, "dailyDetectionCounts": [{"date": "x"}]}},
        [],
    ],
)
def test_an_answer_in_another_shape_raises(station, monkeypatch, answer):
    """A GraphQL error comes back 200, so the status alone would read it as birds."""
    source = station()
    monkeypatch.setattr(fake, "graphql", lambda *_: answer)
    for call in _asked(source):
        with pytest.raises(Unavailable):
            call()


def test_a_station_with_no_birds_reads_as_empty(station):
    quiet = station(rows=[])
    assert quiet.latest() is None
    assert quiet.species_since(6) == []
    assert quiet.species_since(0) == []
    assert quiet.life_list() == []
    assert quiet.last_heard(24) == []


def test_only_birds_come_off_it_and_each_drop_is_said_once(station, caplog):
    now = datetime.now(UTC)
    heard = [BLACKBIRD, BAT, "Sciurus carolinensis", "Human vocal"]
    source = station(rows=[_row(i, now - timedelta(hours=i), name) for i, name in enumerate(heard)])

    with caplog.at_level(logging.INFO, logger="fugleramme.birdweather"):
        assert source.species_since(6) == [(BLACKBIRD, 1)]
        assert source.species_since(0) == [(BLACKBIRD, 1)]
        assert [s.scientific_name for s in source.life_list()] == [BLACKBIRD]
        assert {d.scientific_name for d in source.recent(10)} == {BLACKBIRD}
    assert len(caplog.messages) == 3
    assert all(any(name in line for line in caplog.messages) for name in heard[1:])


def test_a_window_counts_what_it_heard_in_those_hours(station):
    rows = fake.generate(300, 4, fake.BIRDWEATHER_HEARD)
    since = datetime.now(UTC) - timedelta(hours=6)
    heard = Counter(
        row.scientific_name
        for row in rows
        if row.at >= since
        and row.scientific_name not in fake.NON_BIRDS
        and row.scientific_name != fake.GOSHAWK_WAS
    ) + Counter(
        fake.GOSHAWK_IS
        for row in rows
        if row.at >= since and row.scientific_name == fake.GOSHAWK_WAS
    )
    assert station(rows=rows).species_since(6) == sorted(
        heard.items(), key=lambda pair: (-pair[1], pair[0])
    )


def test_the_newest_detections_page_past_birdweather_s_cap(station):
    """`first` stops at 500 without a word, so a longer feed has to page."""
    rows = fake.generate(2000, 2, fake.BIRDWEATHER_HEARD)
    birds = [row.id for row in rows if row.scientific_name not in fake.NON_BIRDS]
    assert len(birds) > 1100  # else this proves nothing
    assert [d.id for d in station(rows=rows).recent(1100)] == birds[:1100]


# -- a reclassified species ---------------------------------------------------
#
# BirdWeather keeps a legacy and a current name as two species with two ids.


def test_a_reclassified_species_is_one_bird(station):
    now = datetime.now(UTC)
    rows = [_row(5 - i, now - timedelta(minutes=i + 1), fake.GOSHAWK_IS) for i in range(2)]
    rows += [_row(3 - i, now - timedelta(days=180, minutes=i), fake.GOSHAWK_WAS) for i in range(3)]
    source = station(rows=rows)

    assert source.species_since(0) == [(fake.GOSHAWK_IS, 5)]
    assert source.species_since(24 * 365) == [(fake.GOSHAWK_IS, 5)]
    assert {d.scientific_name for d in source.recent(10)} == {fake.GOSHAWK_IS}
    (goshawk,) = source.life_list()
    assert goshawk.scientific_name == fake.GOSHAWK_IS
    assert now - goshawk.first_seen > timedelta(days=170)  # the legacy row's first day


# -- days and hours -----------------------------------------------------------


def test_a_bird_first_heard_today_is_dated_to_the_second(station):
    """The record holds UTC days, and "new" is counted in hours. An older bird
    is dated to its day, at noon so the date reads the same in any zone."""
    now = datetime.now(UTC).replace(microsecond=0)
    old = now - timedelta(days=10)
    rows = [
        _row(3, now - timedelta(hours=1), TIT),
        _row(2, now - timedelta(hours=2), TIT),
        _row(1, old, BLACKBIRD),
    ]
    first = {s.scientific_name: s.first_seen for s in station(rows=rows).life_list()}
    assert first[TIT] == now - timedelta(hours=2)
    assert first[BLACKBIRD] == datetime.combine(old.date(), time(12), UTC)


def test_a_station_s_first_day_is_its_utc_one(station):
    """Just after midnight in Oslo is still the day before in UTC, which is how
    the record is dated. Asked from the local date, it leaves that bird out."""
    oslo = ZoneInfo(fake.STATION_ZONE)
    early = datetime.combine(datetime.now(oslo).date() - timedelta(days=10), time(0, 30), oslo)
    rows = [_row(2, datetime.now(UTC) - timedelta(hours=1), TIT), _row(1, early, BLACKBIRD)]
    first = station(rows=rows).life_list()[0]
    assert first.scientific_name == BLACKBIRD
    assert first.first_seen.date() == early.astimezone(UTC).date()


def test_last_heard_orders_by_the_newest_detections_then_by_day(station, monkeypatch):
    now = datetime.now(UTC)
    rows = [
        _row(5, now - timedelta(minutes=5), TIT),
        _row(4, now - timedelta(minutes=30), BLACKBIRD),
        _row(3, now - timedelta(minutes=40), BLACKBIRD),
        _row(2, now - timedelta(days=2), CROW),
        _row(1, now - timedelta(days=3), ROBIN),
    ]
    source = station(rows=rows)
    assert source.last_heard(24) == [TIT, BLACKBIRD]

    monkeypatch.setattr(birdweather, "_PAGE", 1)  # the newest page reaches the tit alone
    assert station(rows=rows).last_heard(0) == [TIT, BLACKBIRD, CROW, ROBIN]


def test_the_newest_detection_is_read_however_long_ago_it_was(station):
    """BirdWeather's `detections` answers the last 24 hours unless given a
    period, which would read a station quiet since the day before as empty."""
    heard = datetime.now(UTC) - timedelta(days=3)
    latest = station(rows=[_row(1, heard, TIT)]).latest()
    assert latest is not None and latest.detected_at == heard.replace(microsecond=0)


# -- several stations ---------------------------------------------------------

OTHER = fake.OTHER_STATION_ID


def test_several_stations_are_one_station(station):
    """A species heard at both is one bird: its counts summed, first heard the
    earliest anywhere, last heard the newest anywhere."""
    now = datetime.now(UTC).replace(microsecond=0)
    rows = [
        _row(6, now - timedelta(minutes=5), ROBIN, OTHER),
        _row(5, now - timedelta(minutes=10), TIT),
        _row(4, now - timedelta(hours=1), ROBIN),
        _row(3, now - timedelta(hours=2), ROBIN),
        _row(2, now - timedelta(days=4), BLACKBIRD),
        _row(1, now - timedelta(days=10), ROBIN, OTHER),
    ]
    both = station(rows=rows, stations=(fake.STATION_ID, OTHER))

    assert both.species_since(6) == [(ROBIN, 3), (TIT, 1)]
    assert both.species_since(0) == [(ROBIN, 4), (TIT, 1), (BLACKBIRD, 1)]
    assert both.latest().id == 6
    assert both.last_heard(24) == [ROBIN, TIT]
    first = {s.scientific_name: s.first_seen for s in both.life_list()}
    assert first[ROBIN].date() == (now - timedelta(days=10)).date()

    alone = station(rows=rows)  # the first station hears none of the other's
    assert alone.species_since(0) == [(ROBIN, 2), (TIT, 1), (BLACKBIRD, 1)]


def test_a_station_birdweather_does_not_have_leaves_the_rest(station):
    """A station deleted or made private is left out rather than taking the
    page with it, but with none left there is nothing to read, not no birds."""
    now = datetime.now(UTC)
    rows = [_row(1, now - timedelta(hours=1), TIT)]
    source = station(rows=rows, stations=(fake.STATION_ID, "404"))
    assert source.species_since(6) == [(TIT, 1)]
    assert list(source.about()) == [fake.STATION_ID, "404"]
    assert source.about()["404"] is None

    for call in _asked(station(rows=rows, stations=("404", "405"))):
        with pytest.raises(NoSuchStation):
            call()


# -- caching ------------------------------------------------------------------


def test_a_repeated_question_is_answered_without_asking_again(detector):
    url, httpd = detector(birdweather=True)
    source = BirdWeatherSource(_page(url))
    first = (source.species_since(6), source.species_since(0), source.recent(5))

    httpd.shutdown()
    httpd.server_close()
    assert (source.species_since(6), source.species_since(0), source.recent(5)) == first


def test_the_whole_record_is_asked_for_rarely_and_its_last_two_days_often(station, monkeypatch):
    """Many frames can share a public station, and its record can run to years."""
    asked: list[str] = []
    answer = fake.graphql

    def record(rows, query, variables):
        if "dailyDetectionCounts" in query:
            asked.append(variables["period"]["from"])
        return answer(rows, query, variables)

    monkeypatch.setattr(fake, "graphql", record)
    now = [0.0]
    monkeypatch.setattr(birdweather, "time", SimpleNamespace(monotonic=lambda: now[0]))
    source = station()
    yesterday = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    source.life_list()
    assert len(asked) == 2 and asked[0] < yesterday == asked[1]

    now[0] += birdweather._TTL + 1
    source.life_list()
    assert asked[2:] == [yesterday]

    now[0] += birdweather._RECORD_TTL
    source.life_list()
    assert asked[3:] == [asked[0], yesterday]


# -- the wrapper --------------------------------------------------------------


def test_the_configured_detector_follows_its_kind(tmp_path, detector):
    birdnet, _one = detector(count=4, seed=0)
    weather, _two = detector(birdweather=True)
    store = SettingsStore(tmp_path / "s.json", Settings(detector_url=birdnet))
    source = Configured(store)
    assert isinstance(source.source, ApiSource)

    store.update(detector_kind=BIRDWEATHER, detector_station=_page(weather))
    assert isinstance(source.source, BirdWeatherSource)
    assert source.species_since(0)
    with pytest.raises(Unavailable):
        source.request("/settings/locales")  # BirdNET-Go's names: a station has none

    store.update(detector_kind=BIRDNET_GO)
    assert isinstance(source.source, ApiSource)
    assert source.base_url == birdnet  # its own address, kept while BirdWeather was read


# -- the fake -----------------------------------------------------------------


def _post(url: str, query: str, **variables) -> tuple[int, bytes]:
    body = json.dumps({"query": query, "variables": variables}).encode()
    request = Request(f"{url}/graphql", data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def test_the_fake_refuses_what_birdweather_refuses(detector):
    url, _httpd = detector(birdweather=True)
    status, body = _post(url, "query ($id: ID!) { station(id: $id) { name } }", id="999")
    assert status == 404 and body.startswith(b"<!DOCTYPE html>")

    fields = " ".join(f"d{i}: detections(first: $first) {{ nodes {{ id }} }}" for i in range(4))
    status, body = _post(url, f"query ($first: Int) {{ {fields} }}", first=1)
    assert status == 200 and "limit 3" in json.loads(body)["errors"][0]["message"]


def test_the_fake_translates_as_birdweather_does(detector):
    """English is the species' commonName, never a translation, and a species
    can lack a locale or be left out of `allSpecies` altogether."""
    url, _httpd = detector(birdweather=True)
    query = (
        "query ($ids: [ID!]!) { allSpecies(ids: $ids) { nodes {"
        " scientificName commonName translations { locale commonName } } } }"
    )
    ids = ["35", "17273", "17078"]  # blackbird, bat, "Human vocal"
    _status, body = _post(url, query, ids=ids)
    nodes = {
        node["scientificName"]: node for node in json.loads(body)["data"]["allSpecies"]["nodes"]
    }
    assert set(nodes) == {BLACKBIRD, BAT}
    blackbird = {t["locale"]: t["commonName"] for t in nodes[BLACKBIRD]["translations"]}
    assert nodes[BLACKBIRD]["commonName"] == "Eurasian Blackbird" and "en" not in blackbird
    assert blackbird["no"] == "Svarttrost"
    assert "zh-SM" not in {t["locale"] for t in nodes[BAT]["translations"]}
